"""BEAT public client API. Importing this package does not launch Julia."""

from .client import EngineWorker, WorkerPool
from .paths import EnginePaths, engine_paths

__version__ = "0.1.0rc1"
__all__ = ["EnginePaths", "EngineWorker", "WorkerPool", "engine_paths"]
