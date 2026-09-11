"""Capture the compiled-system request a Boundary Lab project sends to BEAT.

Run inside a Boundary Lab environment (it imports `blab`) with this package
importable, e.g. from a Boundary Lab checkout:

  PYTHONPATH=<beat_engine checkout>/src .venv/bin/python \\
      <beat_engine checkout>/scripts/capture_boundary_lab_request.py \\
      examples/Multi_region_SAWMOD/Multi_region_SAWMOD.blab.json --out sawmod.json

The request is what `blab project solve` would submit (outputs, excitations,
symmetry, solver options), with the frequencies replaced by --frequencies. Mesh
paths in it are absolute, so it stays tied to that Boundary Lab checkout.
"""

import argparse
import json
from pathlib import Path

from blab.headless import HeadlessSolveSpec, load_headless_project, prepare_headless_solve
from blab.system_contract import system_solve_request_to_dict

DEFAULT_FREQUENCIES = [20.0 * 1000.0 ** (i / 11) for i in range(12)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path, help=".blab.json project")
    parser.add_argument("--frequencies", type=float, nargs="+", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    project = load_headless_project(args.project)
    spec = HeadlessSolveSpec(frequencies_hz=tuple(args.frequencies), raw={"schema_version": 1})
    prepared = prepare_headless_solve(project, spec, backend_id="beat_cpu")
    request = system_solve_request_to_dict(prepared.request)
    args.out.write_text(json.dumps(request))
    print(f"{args.out}: {prepared.solve_kind}, symmetry {project.symmetry}, "
          f"{len(request['excitation_port_ids'])} excitations, {len(request['frequencies_hz'])} frequencies")


if __name__ == "__main__":
    main()
