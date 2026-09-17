import json
import queue
import threading
from pathlib import Path

import pytest

from beat_engine.worker import WorkerPool, WorkerProcess


class Timer:
    def __init__(self, interval, callback, args=()):
        self.interval, self.callback, self.args = interval, callback, args
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.callback(*self.args)


class Process:
    def __init__(self):
        self.stdin = self
        self.stdout = self
        self.lines = queue.Queue()
        self.commands = []
        self.reclaim_started = threading.Event()
        self.stopped = False
        self.reason = "reuse"

    def write(self, text):
        command = json.loads(text)
        self.commands.append(command)
        if command["operation"] == "reclaim":
            self.reclaim_started.set()
        else:
            event = {"type": "completed"}
            if command["operation"] == "solve":
                event["worker_cleanup"] = {"reason": self.reason}
            self.send(event)

    def send(self, event):
        self.lines.put(json.dumps(event) + "\n")

    def flush(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        line = self.lines.get(timeout=3)
        if line is None:
            raise StopIteration
        return line

    def poll(self):
        return 0 if self.stopped else None

    def terminate(self):
        self.stopped = True
        self.lines.put(None)

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr("beat_engine.worker.threading.Timer", Timer)
    worker = WorkerProcess(julia_executable="unused", solver_script=Path("unused"), julia_threads=1, julia_project=None)
    worker._process = Process()
    worker._worker_info = {"operations": ["solve", "bem_field", "reclaim"]}
    worker.configure_idle_cleanup(5000)
    yield worker
    worker.terminate()


def solve(worker, operation="solve"):
    return list(worker.submit(Path("unused"), operation=operation))


def test_idle_after_five_seconds_and_single_reclamation(worker):
    solve(worker)
    timer = worker._idle_timer
    assert timer.interval == 5
    process = worker._process
    process.send({"type": "completed", "worker_cleanup": {"reason": "idle", "requests_since_cleanup": 0}})
    timer.fire()
    assert process.commands[-1]["operation"] == "reclaim"
    assert worker.last_worker_cleanup["reason"] == "idle"
    assert worker._idle_timer is None
    assert not worker._idle_dirty


def test_new_request_cancels_stale_timer_and_field_restarts_idle(worker):
    solve(worker)
    stale = worker._idle_timer
    solve(worker, "bem_field")
    assert stale.cancelled
    assert worker._idle_timer is not stale
    stale.fire()
    assert not worker._process.reclaim_started.is_set()


def test_reservation_spans_preparation_and_is_idempotent(worker):
    solve(worker)
    stale = worker._idle_timer
    release = worker.hold_idle_cleanup()
    stale.fire()
    assert worker._idle_timer is None
    assert not worker._process.reclaim_started.is_set()
    release()
    timer = worker._idle_timer
    release()
    assert worker._idle_timer is timer
    assert worker._idle_holds == 0


@pytest.mark.parametrize("reason", ["interval", "memory_pressure", "memory_unknown", "cancelled", "idle"])
def test_full_cleanup_does_not_schedule_another(worker, reason):
    worker._process.reason = reason
    solve(worker)
    assert worker._idle_timer is None


def test_no_timer_without_advertised_operation(worker):
    worker._worker_info["operations"].remove("reclaim")
    solve(worker)
    assert worker._idle_timer is None


def test_solve_waits_for_inflight_reclamation_without_blocking_caller_reservation(worker):
    solve(worker)
    process = worker._process
    reclaim = threading.Thread(target=worker._idle_timer.fire)
    reclaim.start()
    assert process.reclaim_started.wait(1)
    release = worker.hold_idle_cleanup()  # Non-blocking even while reclamation is active.
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (solve(worker), finished.set()))
    thread.start()
    assert not finished.wait(0.03)
    assert [c["operation"] for c in process.commands] == ["solve", "reclaim"]
    process.send({"type": "completed", "worker_cleanup": {"reason": "idle"}})
    assert finished.wait(1)
    reclaim.join(1)
    thread.join(1)
    assert worker._idle_timer is None
    release()
    assert worker._idle_timer is not None
    assert [c["operation"] for c in process.commands] == ["solve", "reclaim", "solve"]


def test_shutdown_invalidates_pending_timer(worker):
    solve(worker)
    timer = worker._idle_timer
    process = worker._process
    worker.terminate()
    timer.fire()
    assert process.stopped
    assert not process.reclaim_started.is_set()


def test_idle_failure_discards_worker(worker):
    solve(worker)
    process = worker._process
    process.send({"type": "failed", "error": "synthetic failure"})
    worker._idle_timer.fire()
    assert process.stopped
    assert worker._process is None
    assert worker._active_submission is None


def test_pool_reserves_workers_created_during_preparation(worker):
    pool = WorkerPool(lambda **kwargs: worker)
    release = pool.hold_idle_cleanup()
    obtained = pool.get_worker(
        julia_executable="unused", solver_script=Path("unused"), julia_threads=1, julia_project=None
    )
    solve(obtained)
    assert worker._idle_timer is None
    release()
    assert worker._idle_timer is not None
    release()
    pool.shutdown()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_idle_interval(worker, value):
    with pytest.raises(ValueError):
        worker.configure_idle_cleanup(value)


def test_failed_reclamation_is_discarded_before_queued_solve(worker, monkeypatch):
    solve(worker)
    old = worker._process
    cleanup = threading.Thread(target=worker._idle_timer.fire)
    cleanup.start()
    assert old.reclaim_started.wait(1)
    restarted = threading.Event()

    def start_new(startup):
        assert worker._process is None
        assert old.stopped
        worker._process = Process()
        worker._worker_info = {"operations": ["solve", "reclaim"]}
        restarted.set()

    monkeypatch.setattr(worker, "_ensure_started", start_new)
    finished = threading.Event()
    request = threading.Thread(target=lambda: (solve(worker), finished.set()))
    request.start()
    assert not finished.wait(0.03)
    old.send({"type": "failed", "error": "synthetic failure"})
    assert finished.wait(1)
    cleanup.join(1)
    request.join(1)
    assert restarted.is_set()
    assert worker._process is not old
    assert not worker._process.stopped


def test_shutdown_during_reclamation_releases_reader(worker, caplog):
    solve(worker)
    process = worker._process
    cleanup = threading.Thread(target=worker._idle_timer.fire)
    cleanup.start()
    assert process.reclaim_started.wait(1)
    worker.terminate()
    cleanup.join(1)
    assert not cleanup.is_alive()
    assert worker._active_submission is None
    assert worker._idle_timer is None
    assert "idle reclamation failed" not in caplog.text
