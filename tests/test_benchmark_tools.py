import argparse
import array
import base64
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_script("benchmark_worker")
compare = load_script("compare_benchmarks")


def wire(values, shape=None):
    floats = array.array("d", (part for value in values for part in (complex(value).real, complex(value).imag)))
    return [base64.b64encode(floats.tobytes()).decode(), shape if shape is not None else [len(values)]]


def run(values, shape=None):
    return {"outputs": {"pressure@100.000": wire(values, shape)}}


def test_windows_report_preserves_unavailable_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.delattr(os, "uname", raising=False)
    monkeypatch.setattr(benchmark.PeakMemory, "_reader", lambda self: lambda: None)
    with benchmark.PeakMemory(os.getpid()) as memory:
        assert memory.current_mb() is None
    assert memory.peak_mb is None
    args = argparse.Namespace(
        label="test", backend="cpu", precision="float32", request=None, mesh="test.msh", threads="2", frequencies=[]
    )
    out = tmp_path / "run.json"
    benchmark.write_run(out, args, "test", 0, 0, 1, 1, [], None, memory.peak_mb)
    record = json.loads(out.read_text())
    assert record["memory_mb"] == {"before_sweep": None, "peak_sweep": None}
    assert record["host"] == benchmark.platform.machine()
    assert "peak unavailable" in capsys.readouterr().out


def benchmark_result(frequency, values=(1, 2)):
    return {"freq_hz": frequency, "quantities": [
        {"id": name, "quantity": "exterior_pressure", "values": {
            "dtype": "complex128", "shape": [1], "content_base64": wire([value])[0],
        }} for name, value in zip(("probe@near", "probe@far"), values)
    ]}


def test_benchmark_preserves_same_quantity_outputs_and_close_frequencies():
    outputs = {}
    for index, frequency in enumerate((100.0001, 100.0002, 100.0001)):
        outputs.update(benchmark.outputs_of(benchmark_result(frequency), frequency_index=index))
    assert len(outputs) == 6
    metrics = compare.accuracy({"outputs": outputs}, {"outputs": outputs})
    assert set(metrics) == {"exterior_pressure:probe@near", "exterior_pressure:probe@far"}
    changed = dict(outputs)
    key = next(key for key in outputs if "probe@far" in key)
    changed[key] = wire([4])
    assert compare.accuracy({"outputs": changed}, {"outputs": outputs})["exterior_pressure:probe@far"][0] > 0


def test_benchmark_canonicalizes_float32_frequency_without_rounding_identity():
    frequency = 123.456789
    reported = array.array("f", [frequency])[0]
    actual = benchmark.outputs_of(benchmark_result(reported), requested_frequency=frequency)
    expected = benchmark.outputs_of(benchmark_result(frequency), requested_frequency=frequency)
    assert actual == expected
    with pytest.raises(ValueError, match="frequency"):
        benchmark.outputs_of(benchmark_result(124), requested_frequency=frequency)


def test_benchmark_rejects_duplicate_output_identity():
    result = benchmark_result(100)
    result["quantities"].append(result["quantities"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.outputs_of(result)


def test_benchmark_rejects_invalid_output_instead_of_dropping_it():
    result = benchmark_result(100)
    result["quantities"][1]["values"] = None
    with pytest.raises(ValueError, match="Invalid benchmark output"):
        benchmark.outputs_of(result)


def test_benchmark_rejects_missing_sweep_rows(tmp_path):
    args = argparse.Namespace(frequencies=[100, 200])
    with pytest.raises(ValueError, match="zip"):
        benchmark.write_run(tmp_path / "run.json", args, "test", 0, 0, 1, 1, [benchmark_result(100)], None, None)
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("actual", [run([1]), run([1], [2]), run([1, 2], [1, 2]), {"outputs": {}}])
def test_accuracy_rejects_missing_or_truncated_outputs(actual):
    with pytest.raises(ValueError, match="shape|length|quantities"):
        compare.accuracy(actual, run([1, 2]))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_accuracy_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="non-finite"):
        compare.accuracy(run([value]), run([1]))
    with pytest.raises(ValueError, match="non-finite"):
        compare.accuracy(run([1]), run([value]))


def test_accuracy_handles_zero_signals():
    assert compare.accuracy(run([0]), run([1]))["pressure"] == (1.0, math.inf)
    assert compare.accuracy(run([1]), run([0]))["pressure"][0] == math.inf
    assert compare.accuracy(run([0]), run([0]))["pressure"] == (0.0, 0.0)


def test_accuracy_measures_complex_phase():
    relative, db = compare.accuracy(run([1j]), run([1]))["pressure"]
    assert relative == pytest.approx(math.sqrt(2))
    assert db == 0


def test_stage_checks_later_repeats():
    ref = run([1])
    relative, db = compare.stage_accuracy([ref, run([2])], ref)["pressure"]
    assert relative == 1
    assert db == pytest.approx(20 * math.log10(2))
    with pytest.raises(ValueError, match="run 2"):
        compare.stage_accuracy([ref, run([])], ref)


def test_cli_compares_all_repeats_with_unavailable_memory(tmp_path, monkeypatch, capsys):
    for index, value in enumerate([1, 2]):
        record = {
            **run([value]),
            "wall_s": 1.0,
            "commit": "test",
            "backend": "cpu",
            "memory_mb": {"before_sweep": None, "peak_sweep": None},
            "frequencies": [],
        }
        (tmp_path / f"run{index}.json").write_text(json.dumps(record))
    monkeypatch.setattr(
        sys, "argv", ["compare", "--reference", str(tmp_path / "run0.json"), "--stage", f"cpu={tmp_path / 'run*.json'}"]
    )
    compare.main()
    output = capsys.readouterr().out
    assert "memory  unavailable" in output
    assert "pressure 1.0e+00 rel, 6.021 dB" in output


def test_cli_rejects_invalid_later_repeat(tmp_path, monkeypatch):
    reference = {
        **run([1]),
        "wall_s": 1.0,
        "commit": "test",
        "backend": "cpu",
        "memory_mb": {"before_sweep": None, "peak_sweep": None},
        "frequencies": [],
    }
    (tmp_path / "run0.json").write_text(json.dumps(reference))
    (tmp_path / "run1.json").write_text(json.dumps({**reference, "outputs": {}}))
    monkeypatch.setattr(
        sys, "argv", ["compare", "--reference", str(tmp_path / "run0.json"), "--stage", f"cpu={tmp_path / 'run*.json'}"]
    )
    with pytest.raises(SystemExit) as exc:
        compare.main()
    assert exc.value.code == 2
