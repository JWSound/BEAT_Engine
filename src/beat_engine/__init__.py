"""BEAT public client API. Importing this package does not launch Julia."""

from .client import EngineWorker
from .paths import EnginePaths, engine_paths
from .worker import WorkerPool

__version__ = "0.1.0rc1"
__all__ = ["EnginePaths", "EngineWorker", "WorkerPool", "engine_paths"]
