# ⚠️ WORK IN PROGRESS – not finished, not integrated, do not use for production results

This directory holds an **experimental** hierarchical-matrix (H-matrix, ACA) path for the
Burton–Miller BEM operators of the coupled FEM/BEM solver. It is **not** wired into the
engine: Boundary Lab and the normal BEAT solve paths are unchanged by this branch.

Status (2026-11-21):

- Exact-entry oracle for A (Galerkin Burton–Miller, P1/DP0, symmetry images), checked
  against the engine's dense matrix to ~1e-15.
- Entry kernels on the GPU (KernelAbstractions: CPU, Apple Metal Float32, CUDA), batched
  ACA, exact Float64 near-field entries, H-matrix of the right-hand operator C.
- Coupled solve (FEM + BEM + transducers) with GMRES and the bordered near-field
  preconditioner from `src/BeatEngineFmm.jl`, checked offline against the engine's dense
  solution on one real project (development dump hook in `patches/`, applied only for a
  test run and reverted).
- Missing: integration into the engine solve path, avoiding the dense assembly, field
  evaluation from the H path, CUDA timing qualification, sweep handling.

Measurements, decisions and open problems: `docs/FMM_PLAN.md`.
Numbers in that file are development measurements, not release benchmarks.

Layout:
- `hmat_core.jl`, `hmat_fast.jl`, `hmat_ka.jl`, `hmat_c.jl` – H-matrix code.
- `tests/` – standalone test drivers (run from a work directory holding the engine's
  `src/` and `tests/`, see the scripts in `mac_scripts/`).
- `patches/hmat_dump_hook.diff` – env-gated development hook (`BLAB_HMAT_DUMP`), not applied.
- `mac_scripts/` – the scripts used for the Apple Silicon test runs (paths are local).
