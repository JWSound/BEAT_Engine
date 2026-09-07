"""Model-independent physical-system worker with version/capability negotiation."""

import json
from pathlib import Path

from .beat_contract.worker import negotiate_submission, validate_worker_event, validate_worker_ready
from .paths import engine_paths
from .worker import WorkerPool as TransportWorkerPool
from .worker import WorkerProcess


class EngineWorker(WorkerProcess):
    def _accept_ready(self, event: dict) -> None:
        if isinstance(event.get("protocol"), dict) or self.solver_script.resolve() == engine_paths().system_solver:
            validate_worker_ready(event)
        super()._accept_ready(event)

    def _prepare_submission(self, request_path: Path, operation: str) -> dict:
        command = super()._prepare_submission(request_path, operation)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        info = self._worker_info or {}
        protocol = info.get("protocol")
        negotiated = isinstance(protocol, dict) and protocol.get("name") == "beat-worker"
        if operation == "bem_field" or "compiled_system" in request or negotiated:
            command.update(negotiate_submission(info, request, operation))
        return command

    def _accept_event(self, event: dict) -> None:
        protocol = (self._worker_info or {}).get("protocol")
        if isinstance(protocol, dict) and protocol.get("name") == "beat-worker":
            validate_worker_event(event)


class WorkerPool(TransportWorkerPool):
    """Public pool defaults to version-negotiated workers."""

    def __init__(self, factory=EngineWorker):
        super().__init__(factory)
