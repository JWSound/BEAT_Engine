"""Locations of runtime assets in an installed or editable BEAT package."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnginePaths:
    root: Path
    project: Path
    system_solver: Path
    source_solver: Path


def engine_paths(backend: str = "cpu") -> EnginePaths:
    if backend not in {"cpu", "cuda", "rocm", "metal"}:
        raise ValueError(f"Unsupported BEAT backend: {backend}")
    root = Path(__file__).resolve().parent
    project = root / ("julia_local" if backend == "cpu" else f"julia_{backend}")
    return EnginePaths(root, project, root / "julia_local/coupled_solver.jl", root / "julia_local/solver.jl")
