"""Packed meshes and inline worker requests have an independent public contract."""
import base64
import copy
import json
import struct
from pathlib import Path

import pytest
from test_worker_negotiation import fake_worker

from beat_engine.beat_contract import validate_solve_request
from beat_engine.beat_contract.mesh import validate_mesh_data
from beat_engine.beat_contract.worker import WorkerCompatibilityError, negotiate_submission

CONTRACT = Path(__file__).resolve().parents[1] / "src/beat_engine/beat_contract"


def packed(dtype, shape, values):
    fmt = "d" if dtype == "<f8" else "q"
    return {"dtype": dtype, "shape": shape, "data": base64.b64encode(struct.pack("<" + fmt * len(values), *values)).decode()}


def mesh_payload():
    return {"schema_version": 1, "points": packed("<f8", [3, 3], [0., 0, 0, 1, 0, 0, 0, 1, 0]),
            "cells": [{"type": "triangle", "connectivity": packed("<i8", [1, 3], [0, 1, 2]),
                       "physical_tags": packed("<i8", [1], [2])}], "physical_names": {"radiator": [2, 2]}}


def request_and_ready():
    request = json.loads((CONTRACT / "example-exterior-request.json").read_text())
    mesh = request["compiled_system"]["meshes"][0]
    mesh["file"] = ""
    mesh["mesh_data"] = mesh_payload()
    ready = json.loads((CONTRACT / "worker-v1.json").read_text())
    ready["backends"] = {"cpu": {"available": True}}
    return request, ready


def test_packed_mesh_validates_without_numerical_dependencies():
    request, ready = request_and_ready()
    validate_solve_request(request)
    negotiate_submission(ready, request, "solve")
    ready["contracts"].pop("mesh_data")
    with pytest.raises(WorkerCompatibilityError, match="mesh_data"):
        negotiate_submission(ready, request, "solve")
    request["compiled_system"]["meshes"][0]["file"] = "ambiguous.msh"
    with pytest.raises(ValueError, match="either file"):
        validate_solve_request(request)


@pytest.mark.parametrize("damage", ["bytes", "indices", "type", "version", "nan", "names"])
def test_invalid_buffers_are_rejected(damage):
    mesh = mesh_payload()
    if damage == "bytes":
        mesh["points"]["data"] = "bad"
    elif damage == "indices":
        mesh["cells"][0]["connectivity"] = packed("<i8", [1, 3], [0, 1, 3])
    elif damage == "type":
        mesh["cells"][0]["type"] = []
    elif damage == "version":
        mesh["schema_version"] = True
    elif damage == "names":
        mesh["physical_names"] = {}
    else:
        mesh["points"] = packed("<f8", [3, 3], [float("nan")] * 9)
    with pytest.raises(ValueError):
        validate_mesh_data(mesh)


def test_inline_transport_negotiates_before_sending(tmp_path):
    request, ready = request_and_ready()
    worker, received = fake_worker(tmp_path, ready)
    try:
        assert list(worker.submit(request))[-1]["type"] == "completed"
        command = json.loads(received.read_text())
        assert command["request_inline"] == request
        assert "request" not in command
    finally:
        worker.terminate()


def test_old_worker_rejects_inline_before_sending(tmp_path):
    request, ready = request_and_ready()
    ready = copy.deepcopy(ready)
    ready.pop("request_transports")
    worker, received = fake_worker(tmp_path, ready)
    try:
        with pytest.raises(RuntimeError, match="inline"):
            worker.submit(request)
        assert not received.exists()
    finally:
        worker.terminate()
