"""Experimental NumPy/SciPy consumer of a continuous interior in a .blabsp archive.

evaluate(frequency_hz, surface_pressure_pa, port_volts) returns (q, velocity, current).
Supply one load case per column and total exterior surface pressure in package node order.
"""

import base64
import hashlib
import json
import zipfile

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag, solve
from scipy.sparse.linalg import splu


def _matrix(entry):
    shape = tuple(entry["shape"])
    if entry["storage"] == "coo":
        return sparse.csc_matrix((entry["values"], (entry["rows"], entry["columns"])), shape=shape)
    if entry["storage"] != "f64le-column-major":
        raise ValueError("Unsupported continuous ROM matrix encoding")
    return np.frombuffer(base64.b64decode(entry["base64"], validate=True), dtype="<f8").reshape(shape, order="F")


class ContinuousSpeaker:
    def __init__(self, path):
        with zipfile.ZipFile(path) as archive:
            checksums = json.loads(archive.read("checksums.json"))

            def read_json(name):
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != checksums[name]:
                    raise ValueError(f"Checksum mismatch for {name}")
                return json.loads(data)

            manifest = read_json("manifest.json")
            description = (
                manifest["files"].get("experimental_continuous_interior") or manifest["files"]["coupled_model"]
            )
            model = read_json(description["path"])
        if model["metadata"]["format"] not in {"experimental-continuous-speaker-v1", "continuous-speaker-v1"}:
            raise ValueError("Unsupported continuous interior format")
        if model["metadata"]["phasor"] != "exp(+i omega t)":
            raise ValueError("Unsupported phasor convention")
        self.ports = model["ports"]
        self.drivers = model["drivers"]
        self.band = model["band"]
        self.c, self.rho = model["c"], model["rho"]
        self.points_m = np.array(model["geometry"]["points"])
        self.triangles = np.array(model["geometry"]["triangles"])
        self.matrices = {key: _matrix(value) for key, value in model["matrices"].items()}
        self.walls = [
            (w["thickness_m"], w["flow_resistivity_pa_s_per_m2"], _matrix(w["matrix"])) for w in model["walls"]
        ]
        self.gram = splu(self.matrices["gram"].astype(complex))

    def evaluate(self, frequency_hz, surface_pressure_pa, port_volts):
        f = float(frequency_hz)
        if not self.band[0] <= f <= self.band[1]:
            raise ValueError(f"Frequency outside {self.band} Hz")
        p, u = np.asarray(surface_pressure_pa), np.asarray(port_volts)
        if p.ndim != 2 or u.ndim != 2 or p.shape != (len(self.points_m), u.shape[1]) or len(u) != len(self.ports):
            raise ValueError("Expected node pressures and port voltages with one column per load case")

        m = self.matrices
        s = 2j * np.pi * f
        k = 2 * np.pi * f / self.c
        scale = -self.rho * s
        a = m["K"] - k**2 * (m["M"] - 1j * m["loss"])
        for thickness, resistivity, wall in self.walls:
            ratio = f / resistivity
            impedance = self.rho * self.c * (1 + (0.070 - 0.107j) * ratio**-0.632)
            porous_k = k * (1 + (0.109 - 0.160j) * ratio**-0.618)
            admittance = 1j * np.tan(porous_k * thickness) / impedance
            a -= scale * admittance * wall

        mechanical, electrical = [], []
        for d in self.drivers:
            stiffness = 1 / d["cms_m_per_n"]
            rear = d["lumped_sealed_rear_chamber"]
            if rear is not None:
                stiffness += self.rho * self.c**2 * rear["projected_area_m2"] ** 2 / rear["volume_m3"]
            mechanical.append(d["rms_n_s_per_m"] + s * d["mmd_kg"] + stiffness / s)
            coil = d["semi_inductance"]
            if coil is None:
                electrical.append(d["re_ohm"] + s * d["le_h"])
            else:
                parallel = 1 / coil["rss_ohm"] + 1 / (s * coil["le_h"]) + 1 / (np.sqrt(s) * coil["ke_semi_h"])
                electrical.append(coil["re_prime_ohm"] + s * coil["leb_h"] + 1 / parallel)
        bl = np.diag([d["bl_n_per_a"] for d in self.drivers])

        ng = self.gram.shape[0]
        gamma = m["trace"] @ p
        inputs = np.hstack((-a[ng:, :ng] @ gamma, scale * m["surface"][ng:]))
        free, forced = np.split(solve(a[ng:, ng:], inputs), [p.shape[1]], axis=1)
        system = block_diag(np.diag(mechanical) - m["force"][ng:].T @ forced, np.diag(electrical))
        nd = len(self.drivers)
        system[:nd, nd:], system[nd:, :nd] = -bl, bl
        load = m["force"][ng:].T @ free + m["force"][:ng].T @ gamma - m["bem_force"].T @ p
        velocity, current = np.split(solve(system, np.vstack((load, m["drive"] @ u))), [nd])
        interior = free + forced @ velocity
        residual = a[:ng, :ng] @ gamma + a[:ng, ng:] @ interior - scale * m["surface"][:ng] @ velocity
        q = m["flux"] @ self.gram.solve(residual) + scale * (m["bem_velocity"] @ velocity)
        return q, velocity, current


def check_sampled_traces(path, reference_voltage):
    """Replay each fallback sample and compare its complex outward normal derivative."""
    import io

    speaker = ContinuousSpeaker(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        fixed = manifest["files"]["fixed_sources"]["path"]
        with np.load(io.BytesIO(archive.read(fixed))) as traces:
            p, q = traces["pressure_pa"], traces["normal_derivative_pa_per_m"]
            np.testing.assert_array_equal(speaker.triangles, traces["triangles"])
            np.testing.assert_allclose(speaker.points_m, traces["points_m"], atol=1e-8, rtol=0)
    if speaker.ports != manifest["excitation_port_ids"]:
        raise ValueError("Fallback traces and continuous interior disagree on port order")
    voltage = reference_voltage * np.eye(len(speaker.ports))
    rows = []
    for index, f in enumerate(manifest["frequencies_hz"]):
        estimated, _, _ = speaker.evaluate(f, p[index].T, voltage)
        ref = q[index].T
        rows.append(
            {"frequency_hz": f, "relative_flux_error": float(np.linalg.norm(estimated - ref) / np.linalg.norm(ref))}
        )
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare the continuous ROM against sampled package traces.")
    parser.add_argument("package")
    parser.add_argument("--reference-voltage", type=float, default=2.83)
    args = parser.parse_args()
    print(json.dumps(check_sampled_traces(args.package, args.reference_voltage), indent=2))
