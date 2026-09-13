"""Locations of runtime assets in an installed or editable BEAT package."""

from dataclasses import dataclass
from pathlib import Path

from .backends import backend_info


@dataclass(frozen=True)
class EnginePaths:
    root: Path
    project: Path
    system_solver: Path
    source_solver: Path


def engine_paths(backend: str = "cpu") -> EnginePaths:
    info = backend_info(backend)
    root = Path(__file__).resolve().parent
    project = root / info.project_directory
    return EnginePaths(root, project, root / "julia_local/coupled_solver.jl", root / "julia_local/solver.jl")
