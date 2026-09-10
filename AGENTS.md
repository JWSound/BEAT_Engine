# BEAT Engine contributor guide

Python public runtime code uses the standard library and must not import Boundary
Lab models, Qt, NumPy, or application preferences. Numerical source is under
src/beat_engine/julia_local; CPU/CUDA/ROCm environments are adjacent packages.
Preserve complex per-excitation results and explicit version negotiation.

Run python -m pytest and python -m ruff check src/beat_engine/*.py
src/beat_engine/beat_contract tests for Python changes. Run the standalone Julia
reference gate for numerical changes. Keep LICENSE, fixture hashes, and the
extraction commit map. Never regenerate numerical baselines just to pass a test.
