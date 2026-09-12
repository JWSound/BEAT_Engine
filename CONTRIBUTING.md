# Contributing

`AGENTS.md` is the contributor guide, for people and coding agents alike. In
short:

- Python changes: `python -m pytest` and
  `python -m ruff check src/beat_engine/*.py src/beat_engine/beat_contract tests`.
- Numerical changes: the standalone Julia reference gate,
  `julia --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/reference_tests.jl`.
  Never regenerate a numerical baseline to make a test pass.
- A change that alters results goes in its own PR, separate from backend or
  performance work, so it can be reviewed on its own terms.
- Performance, memory or accuracy claims: measure them as in
  `docs/Benchmarking.md` and put the numbers, with their run-to-run range, in
  the PR.
