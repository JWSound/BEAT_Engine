"""Backends supplied by this engine distribution; no runtime or hardware probing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendInfo:
    """Static engine capabilities, not a promise that a device is available.

    IDs are stable across releases. Platforms use Python's sys.platform names.
    Runtime/device and request compatibility remain worker-handshake checks.
    """

    backend_id: str
    label: str
    project_directory: str
    platforms: tuple[str, ...]
    solve_kinds: tuple[str, ...] = ("exterior_bem", "interior_fem", "coupled_fem_bem_lem")
    supports_symmetry: bool = True
    supports_channel_resynthesis: bool = True
    condenses_fem_interior: bool = True


_BACKENDS = (
    BackendInfo("cuda", "BEAT Engine (Nvidia CUDA)", "julia_cuda", ("linux", "win32")),
    BackendInfo("cpu", "BEAT Engine (CPU)", "julia_local", ("linux", "win32", "darwin")),
    BackendInfo("rocm", "BEAT Engine (AMD ROCm)", "julia_rocm", ("linux", "win32")),
    BackendInfo("metal", "BEAT Engine (Apple Metal)", "julia_metal", ("darwin",)),
)


def backend_catalog() -> tuple[BackendInfo, ...]:
    """Return every backend in this installed engine, without launching Julia.

    The catalog is identical on every host, so an unavailable saved selection
    can still be displayed. Inspect the worker announcement at solve time to
    determine whether the configured environment can execute a request.
    """
    return _BACKENDS


def backend_info(backend_id: str) -> BackendInfo:
    """Look up a canonical backend ID; never silently select another backend."""
    for backend in backend_catalog():
        if backend.backend_id == backend_id:
            return backend
    raise ValueError(f"Unsupported BEAT backend: {backend_id}")
