import json
from pathlib import Path

import pytest

from beat_engine.beat_contract.worker import WorkerCompatibilityError, negotiate_submission
from beat_engine.client import EngineWorker

CONTRACT = Path(__file__).resolve().parents[1] / "src/beat_engine/beat_contract"


def test_positive_time_requires_advertised_support():
    ready = json.loads((CONTRACT / "worker-v1.json").read_text())
    ready["backends"] = {"cpu": {"available": True}}
    payload = json.loads((CONTRACT / "example-exterior-request.json").read_text())
    payload.setdefault("solver_options", {})["phasor_convention"] = "exp(+i omega t)"
    ready.pop("phasor_conventions", None)
    with pytest.raises(WorkerCompatibilityError, match="phasor"):
        negotiate_submission(ready, payload, "solve")
    ready["phasor_conventions"] = ["exp(+i omega t)"]
    assert negotiate_submission(ready, payload, "solve")["phasor_convention"] == "exp(+i omega t)"


@pytest.mark.parametrize("actual", [None, "exp(-i omega t)", "unknown"])
def test_worker_rejects_mismatched_complex_results(actual):
    worker = object.__new__(EngineWorker)
    worker._worker_info = {}
    worker._expected_phasor = "exp(+i omega t)"
    event = {"type": "result", "result": {"diagnostics": {}}}
    if actual is not None:
        event["result"]["diagnostics"]["phasor_convention"] = actual
    with pytest.raises(RuntimeError, match="phasor convention mismatch"):
        worker._accept_event(event)


def test_backend_can_withhold_unqualified_convention():
    ready = json.loads((CONTRACT / "worker-v1.json").read_text())
    ready["backends"] = {"rocm": {"available": True, "phasor_conventions": ["exp(-i omega t)"]}}
    payload = json.loads((CONTRACT / "example-exterior-request.json").read_text())
    payload["solver_options"].update(phasor_convention="exp(+i omega t)", bem_backend="rocm")
    with pytest.raises(WorkerCompatibilityError, match="not qualified"):
        negotiate_submission(ready, payload, "solve")
