import json

import pytest

from beat_engine import __version__, engine_paths


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm"])
def test_public_paths_resolve_packaged_assets(backend):
    paths = engine_paths(backend)
    assert (paths.project / "Project.toml").is_file()
    assert paths.system_solver.is_file()
    assert paths.source_solver.is_file()
    info = json.loads((paths.root / "beat_contract/worker-v1.json").read_text())
    assert info["engine"]["version"] == __version__


def test_unsupported_backend_does_not_fall_back():
    with pytest.raises(ValueError, match="Unsupported"):
        engine_paths("metal")
