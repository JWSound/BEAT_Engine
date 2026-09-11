# BEAT Engine contributor guide

Python public runtime code uses the standard library and must not import Boundary
Lab models, Qt, NumPy, or application preferences. Numerical source is under
src/beat_engine/julia_local; CPU/CUDA/ROCm environments are adjacent packages.
Preserve complex per-excitation results and explicit version negotiation.

Run python -m pytest and python -m ruff check src/beat_engine/*.py
src/beat_engine/beat_contract tests for Python changes. Run the standalone Julia
reference gate for numerical changes. Keep LICENSE, fixture hashes, and the
extraction commit map. Never regenerate numerical baselines just to pass a test.

Changes that alter numerical results go in their own PR, never alongside backend
or performance work. Performance claims follow docs/Benchmarking.md: real
projects on each solve path touched, at least 3 interleaved runs with median and
range, per-section timings, peak memory and accuracy against a reference; no
claim inside run-to-run variance. Bound an overlap or offload's gain from
measured timings before building it, and take its on/off choice from
calibration, not a constant measured on one machine.
