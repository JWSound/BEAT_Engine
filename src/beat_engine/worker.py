"""Model-independent Julia worker transport, using only the Python standard library.

Callers supply executable/script paths and a complete child-process environment.
Requests and events are opaque JSON documents; no acoustic or application models
are imported here. WorkerPool owns only workers obtained through that pool.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path


class WorkerProcess:
    def __init__(
        self,
        *,
        julia_executable: str,
        solver_script: Path,
        julia_threads: str | int,
        julia_project: Path | None,
        julia_sysimage: Path | None = None,
        environment: Mapping[str, str] | None = None,
        backend_label: str = "the selected BEAT Engine backend",
        startup_timeout_s: float = 300.0,
    ):
        self.julia_executable = julia_executable
        self.solver_script = solver_script
        self.julia_threads = julia_threads
        self.julia_project = julia_project
        self.julia_sysimage = julia_sysimage
        self.environment = dict(os.environ if environment is None else environment)
        self.environment["JULIA_NUM_THREADS"] = resolve_julia_threads(julia_threads)
        self.backend_label = backend_label
        if startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive.")
        self.startup_timeout_s = startup_timeout_s
        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)
        self._active_submission: _SubmissionToken | None = None
        self._starting: _StartupToken | None = None
        self._terminating = False
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._status_callback: Callable[[str], None] | None = None
        self._worker_info: dict | None = None

    @property
    def worker_info(self) -> dict | None:
        """A copy of the current process's ready announcement, if any."""
        return copy.deepcopy(self._worker_info)

    def _accept_ready(self, event: dict) -> None:
        self._worker_info = copy.deepcopy(event)

    def _prepare_submission(self, request_path: Path, operation: str) -> dict:
        return {"request": str(request_path), "operation": str(operation)}

    def _accept_event(self, event: dict) -> None:
        """Optional protocol validation before exposing a submission event."""

    def submit(
        self,
        request_path: Path,
        *,
        status_callback: Callable[[str], None] | None = None,
        operation: str = "solve",
    ) -> Iterator[dict]:
        with self._available:
            while self._active_submission is not None or self._starting is not None or self._terminating:
                self._available.wait()
            startup = _StartupToken()
            self._starting = startup
            self._status_callback = status_callback
        try:
            self._ensure_started(startup)
            with self._available:
                self._check_startup(startup)
                command = self._prepare_submission(request_path, operation)
                process = self._process
                if process is None or process.stdin is None:
                    raise RuntimeError("Warm BEAT Engine solver did not provide stdin.")
                self._emit_status("Submitting solve request" if operation == "solve" else "Submitting field request")
                process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
                process.stdin.flush()
                token = _SubmissionToken()
                self._active_submission = token
                return _SubmissionEvents(self, process, token)
        finally:
            with self._available:
                if self._starting is startup:
                    self._starting = None
                    if self._active_submission is None:
                        self._status_callback = None
                    self._available.notify_all()

    def ensure_started(self, *, status_callback: Callable[[str], None] | None = None) -> None:
        with self._available:
            while self._active_submission is not None or self._starting is not None or self._terminating:
                self._available.wait()
            startup = _StartupToken()
            self._starting = startup
            previous_callback = self._status_callback
            self._status_callback = status_callback
        try:
            self._ensure_started(startup)
        finally:
            with self._available:
                if self._starting is startup:
                    self._starting = None
                    self._status_callback = previous_callback
                    self._available.notify_all()

    def terminate(self) -> None:
        self._terminate()

    def _terminate(self, *, expected_startup: _StartupToken | None = None, timeout: bool = False) -> None:
        with self._available:
            while self._terminating:
                self._available.wait()
            if expected_startup is not None and self._starting is not expected_startup:
                return
            if timeout and expected_startup is not None and expected_startup.ready:
                return
            self._terminating = True
            startup = self._starting
            if startup is not None:
                startup.invalidated = True
                startup.reason = "timed out" if timeout else "terminated"
            self._starting = None
            token = self._active_submission
            if token is not None:
                token.invalidated = True
            self._active_submission = None
            self._status_callback = None
            process = self._detach_process()
            self._available.notify_all()
        try:
            self._stop_process(process)
        finally:
            with self._available:
                self._terminating = False
                self._available.notify_all()

    def _discard_expected_process(self, expected: subprocess.Popen[str]) -> None:
        with self._available:
            process = self._detach_process() if self._process is expected else None
        self._stop_process(process)

    def _detach_process(self) -> subprocess.Popen[str] | None:
        process = self._process
        self._process = None
        self._worker_info = None
        return process

    @staticmethod
    def _stop_process(process: subprocess.Popen[str] | None) -> None:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=2.0)

    @staticmethod
    def _check_startup(startup: _StartupToken) -> None:
        if startup.invalidated:
            raise RuntimeError(f"BEAT Engine worker startup {startup.reason}.")

    def _ensure_started(self, startup: _StartupToken) -> None:
        with self._available:
            self._check_startup(startup)
            if self._process is not None and self._process.poll() is None:
                self._emit_status("BEAT Engine ready")
                return

            self._stderr_lines.clear()
            self._worker_info = None
            command = julia_worker_command(
                self.julia_executable,
                self.solver_script,
                julia_project=self.julia_project,
                julia_sysimage=self.julia_sysimage,
            )
            self._emit_status("Initializing BEAT Engine")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.solver_script.parent),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=self.environment,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Julia executable was not found. Configure its executable path or add Julia to PATH."
                ) from exc
            self._process = process
            self._stderr_thread = threading.Thread(target=self._collect_stderr, args=(process,), daemon=True)
            self._stderr_thread.start()

        timer = threading.Timer(
            self.startup_timeout_s, self._terminate, kwargs={"expected_startup": startup, "timeout": True}
        )
        timer.daemon = True
        timer.start()
        try:
            for event in self._read_events(process):
                with self._available:
                    self._check_startup(startup)
                    event_type = str(event.get("type", ""))
                    if event_type == "ready":
                        self._accept_ready(event)
                        startup.ready = True
                        self._emit_status("BEAT Engine ready")
                        return
                if event_type == "failed":
                    self._discard_expected_process(process)
                    raise RuntimeError(
                        format_julia_error(
                            str(event.get("error", "BEAT Engine solver failed during startup.")),
                            julia_project=self.julia_project,
                            backend_label=self.backend_label,
                        )
                    )
            with self._available:
                self._check_startup(startup)
            raise RuntimeError(self._process_error("Warm BEAT Engine solver ended before startup completed."))
        except Exception:
            self._discard_expected_process(process)
            with self._available:
                self._check_startup(startup)
            raise
        finally:
            timer.cancel()

    def _iter_events_for_submission(self, process: subprocess.Popen[str], token: _SubmissionToken) -> Iterator[dict]:
        terminal = False
        try:
            for event in self._read_events(process):
                if token.invalidated:
                    raise RuntimeError("BEAT Engine worker was terminated during submission.")
                self._accept_event(event)
                terminal = str(event.get("type", "")) in {"completed", "cancelled", "failed"}
                yield event
                if terminal:
                    return
            raise RuntimeError(self._process_error("Warm BEAT Engine solver ended before job completion."))
        finally:
            # Unread events belong to this job and cannot become the next
            # submission's results. Restart and renegotiate after abandonment.
            self._finish_submission(token, discard=not terminal)

    def _finish_submission(self, token: _SubmissionToken, *, discard: bool) -> None:
        with self._available:
            if self._active_submission is not token:
                return
            self._active_submission = None
            self._status_callback = None
            process = self._detach_process() if discard else None
            # Keep a new submission from starting until the abandoned child has
            # actually stopped, including its stderr collection.
            self._stop_process(process)
            self._available.notify_all()

    def _read_events(self, process: subprocess.Popen[str] | None = None) -> Iterator[dict]:
        process = self._process if process is None else process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            parse_started = time.perf_counter()
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                yield {"type": "status", "message": text}
                continue
            if isinstance(event, dict):
                if str(event.get("type", "")) == "result":
                    event["_transport"] = {
                        "julia_stdout_bytes": len(text.encode("utf-8")),
                        "python_julia_json_parse_s": time.perf_counter() - parse_started,
                    }
                yield event

        exit_code = process.wait()
        with self._available:
            if self._process is process:
                self._detach_process()
        if exit_code != 0:
            raise RuntimeError(self._process_error(f"Warm BEAT Engine solver exited with code {exit_code}."))

    def _collect_stderr(self, process: subprocess.Popen[str] | None) -> None:
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            if self._process is not process:
                return
            text = line.strip()
            if text:
                self._stderr_lines.append(text)
                self._emit_status(text)

    def _process_error(self, fallback: str) -> str:
        detail = "\n".join(self._stderr_lines[-10:])
        message = f"{fallback}\n{detail}" if detail else fallback
        return format_julia_error(
            message,
            julia_project=self.julia_project,
            detection_text="\n".join(self._stderr_lines),
            backend_label=self.backend_label,
        )

    def _emit_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)


class _SubmissionToken:
    def __init__(self) -> None:
        self.invalidated = False


class _StartupToken:
    def __init__(self) -> None:
        self.invalidated = False
        self.reason = "terminated"
        self.ready = False


class _SubmissionEvents(Iterator[dict]):
    """Closeable stream that owns one submission, even before its first read."""

    def __init__(self, worker: WorkerProcess, process: subprocess.Popen[str], token: _SubmissionToken):
        self._worker = worker
        self._token = token
        self._events = worker._iter_events_for_submission(process, token)
        self._closed = False

    def __iter__(self) -> _SubmissionEvents:
        return self

    def __next__(self) -> dict:
        if self._closed:
            raise StopIteration
        if self._token.invalidated:
            self.close()
            raise RuntimeError("BEAT Engine worker was terminated during submission.")
        try:
            event = next(self._events)
        except BaseException:
            self.close()
            if self._token.invalidated:
                raise RuntimeError("BEAT Engine worker was terminated during submission.") from None
            raise
        if self._token.invalidated:
            self.close()
            raise RuntimeError("BEAT Engine worker was terminated during submission.")
        if str(event.get("type", "")) in {"completed", "cancelled", "failed"}:
            # A terminal event releases ownership immediately; callers need not
            # request one extra item or explicitly close the exhausted stream.
            self.close()
        return event

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._events.close()
            except ValueError as exc:
                # A different thread may be blocked in next(). Stopping the
                # process below wakes it; that reader runs the generator's
                # finally block itself.
                if str(exc) != "generator already executing":
                    raise
                self._token.invalidated = True
        finally:
            # Closing an unstarted generator does not run its finally block.
            self._worker._finish_submission(self._token, discard=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def format_julia_error(
    message: str,
    *,
    julia_project: str | Path | None,
    backend_label: str = "the selected BEAT Engine backend",
    detection_text: str | None = None,
) -> str:
    if julia_project is None:
        return message

    text = f"{detection_text or message}\n{message}".lower()
    missing_dependency_markers = (
        "argumenterror: package",
        "not found in current path",
        "run `import pkg; pkg.add",
        "could not load project",
        "failed to precompile",
    )
    julia_load_markers = (
        "loading.jl",
        "require(into::module",
        "require(uuidkey::base.pkgid",
    )
    cuda_load_markers = (
        "cuda.jl could not be loaded",
        "package cuda",
        "using cuda",
        "import cuda",
    )
    rocm_load_markers = (
        "amdgpu.jl could not be loaded",
        "package amdgpu",
        "using amdgpu",
        "import amdgpu",
    )
    metal_load_markers = (
        "metal.jl could not be loaded",
        "package metal",
        "using metal",
        "import metal",
    )
    looks_like_dependency_error = any(marker in text for marker in missing_dependency_markers)
    looks_like_julia_load_error = any(marker in text for marker in julia_load_markers)
    looks_like_cuda_error = any(marker in text for marker in cuda_load_markers)
    looks_like_rocm_error = any(marker in text for marker in rocm_load_markers)
    looks_like_metal_error = any(marker in text for marker in metal_load_markers)
    if not (
        looks_like_dependency_error
        or looks_like_julia_load_error
        or looks_like_cuda_error
        or looks_like_rocm_error
        or looks_like_metal_error
    ):
        return message

    project_path = Path(julia_project)
    install_command = f'julia --project={project_path} -e "using Pkg; Pkg.instantiate()"'
    return (
        f"BEAT Engine could not load the Julia dependencies for {backend_label}.\n\n"
        "This usually means the selected BEAT Engine Julia environment has not been installed yet. "
        "To install that environment, run:\n\n"
        f"{install_command}\n\n"
        f"Julia reported:\n{message}"
    )


def resolve_julia_threads(julia_threads: str | int = "auto") -> str:
    if isinstance(julia_threads, int):
        return str(max(1, julia_threads))

    text = str(julia_threads or "auto").strip().lower()
    if text == "auto":
        return str(os.cpu_count() or 1)

    try:
        return str(max(1, int(text)))
    except ValueError:
        return str(os.cpu_count() or 1)


def julia_command(
    julia_executable: str,
    solver_script: Path,
    request_path: Path,
    *,
    julia_project: Path | None,
    julia_sysimage: Path | None = None,
) -> list[str]:
    command = [julia_executable]
    if julia_sysimage is not None:
        command.append(f"--sysimage={julia_sysimage}")
    if julia_project is not None:
        command.append(f"--project={julia_project}")
        command.append("--startup-file=no")
    command.extend([str(solver_script), "--request", str(request_path)])
    return command


def julia_worker_command(
    julia_executable: str,
    solver_script: Path,
    *,
    julia_project: Path | None,
    julia_sysimage: Path | None = None,
) -> list[str]:
    command = [julia_executable]
    if julia_sysimage is not None:
        command.append(f"--sysimage={julia_sysimage}")
    if julia_project is not None:
        command.append(f"--project={julia_project}")
        command.append("--startup-file=no")
    command.extend([str(solver_script), "--worker"])
    return command


class WorkerPool:
    """Reuse workers only when their execution settings and environments match."""

    def __init__(self, factory: Callable[..., WorkerProcess] = WorkerProcess):
        self._factory = factory
        self._lock = threading.Lock()
        self._workers: dict[tuple, WorkerProcess] = {}

    def get_worker(
        self,
        *,
        julia_executable: str,
        solver_script: Path,
        julia_threads: str | int,
        julia_project: Path | None,
        julia_sysimage: Path | None = None,
        environment: Mapping[str, str] | None = None,
        backend_label: str = "the selected BEAT Engine backend",
        startup_timeout_s: float = 300.0,
    ) -> WorkerProcess:
        threads = resolve_julia_threads(julia_threads)
        env = dict(os.environ if environment is None else environment)
        env["JULIA_NUM_THREADS"] = threads
        key = (
            julia_executable,
            str(solver_script.resolve()),
            "" if julia_project is None else str(julia_project.resolve()),
            "" if julia_sysimage is None else str(julia_sysimage.resolve()),
            threads,
            tuple(sorted(env.items())),
            backend_label,
            startup_timeout_s,
        )
        with self._lock:
            worker = self._workers.get(key)
            if worker is None:
                worker = self._factory(
                    julia_executable=julia_executable,
                    solver_script=solver_script,
                    julia_threads=threads,
                    julia_project=julia_project,
                    julia_sysimage=julia_sysimage,
                    environment=env,
                    backend_label=backend_label,
                    startup_timeout_s=startup_timeout_s,
                )
                self._workers[key] = worker
            return worker

    def shutdown(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.terminate()
