# Contributing to BEAT Engine

Branch from `main` and submit a scoped pull request back to `main`. Forks are welcome.
There is no permanent dev branch. Use draft PRs for early feedback; link related
Boundary Lab or Deploy PRs when changing a shared contract.

Python 3.11+ and Julia 1.12 are required. From a virtual environment:

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src/beat_engine/*.py src/beat_engine/beat_contract tests scripts/release_tools.py
python -m ruff format --check pyproject.toml scripts src tests
python -m beat_engine instantiate --backend cpu
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/runtests.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/reference_tests.jl
```

Public Python runtime code uses the standard library. Keep application UI and
preferences in the application repositories. Preserve complex results, provenance
and backwards-compatible worker contracts. Do not regenerate numerical baselines
to make tests pass. Numerical-result changes must have separate PRs from backend
or performance work. See AGENTS.md and docs/Benchmarking.md.

CPU CI is available to all contributors. Accelerator qualification is a separate
maintainer-triggered workflow on trusted hardware; contributors need not own a GPU.
Never run unreviewed fork code on persistent self-hosted runners.

See [release process](docs/development.md) for versioning and application adoption.
