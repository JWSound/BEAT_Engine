import json

import pytest

from beat_engine import __version__, engine_paths


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm", "metal"])
def test_public_paths_resolve_packaged_assets(backend):
    paths = engine_paths(backend)
    assert (paths.project / "Project.toml").is_file()
    assert paths.system_solver.is_file()
    assert paths.source_solver.is_file()
    info = json.loads((paths.root / "beat_contract/worker-v1.json").read_text())
    assert info["engine"]["version"] == __version__


def test_unsupported_backend_does_not_fall_back():
    with pytest.raises(ValueError, match="Unsupported"):
        engine_paths("opencl")


def test_public_pool_defaults_to_negotiated_engine_workers():
    from beat_engine import EngineWorker, WorkerPool

    paths = engine_paths()
    pool = WorkerPool()
    try:
        worker = pool.get_worker(
            julia_executable="julia", solver_script=paths.system_solver, julia_threads=2, julia_project=paths.project
        )
        assert isinstance(worker, EngineWorker)
    finally:
        pool.shutdown()
