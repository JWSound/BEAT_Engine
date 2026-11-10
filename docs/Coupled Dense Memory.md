# Coupled dense memory on Metal (condensed solver)

Per frequency the condensed coupled solve used to hold, at its peak (N_B BEM vertices,
F ≈ 2 N_B BEM faces, n ≈ N_B + 2·transducers the dense order under flux elimination):

| Stage | Before | Now (defaults) |
|---|---|---|
| Combined BEM assembly (A, C) | A (8 N_B²) + C (8 N_B·F) on shared storage | A + C on the flux-carrying columns only (8 N_B·F_active) |
| Host combine | + host copy of A (8 N_B²) + negated C copy (8 N_B·F) | no copies: A and −C stay on the device-backed host views until consumed |
| Dense system + factor | ComplexF64 matrix (16 n²) + ComplexF32 factor (8 n²); F64 LU fallback +16 n² | ComplexF32 store + border corrections (8 n² + 16 n·w) + factor (8 n²); with spill the store is an unlinked scratch file streamed by panels (8 n² resident) |
| Refinement fallback | ComplexF64 LU of the whole system | GMRES-IR (Float32 LU preconditioner applied in Float64) first |

`w` is the border width: the non-BEM unknowns plus the BEM columns that the interface
elimination writes Float64 data into (|Γ|).

## Equivalence

* **Compact C** (`BLAB_METAL_COUPLED_COMPACT_FLUX`): the pair gather, singular write-back and identity
  scatter skip inactive columns; the retained columns are bit-identical to the full assembly
  (`scripts/validate_metal_coupled_combined.jl`, `check_compact_columns`). Products with a flux that is
  nonzero outside the active set raise an error instead of dropping it.
* **Split store** (`BLAB_COUPLED_DENSE_SPLIT`): writes accumulate in Float64 exactly as before; the
  `ComplexF32` store equals `ComplexF32.(M)` and the reconstruction equals `M` bit for bit
  (`tests/split_dense_tests.jl`). The factor is the one `RefinedDenseLU` computed; only the summation
  order of the residual product differs. The refinement acceptance test is unchanged.
* **Spill** (`BLAB_COUPLED_DENSE_SPILL`): same arithmetic, the residual store is streamed from a scratch file
  (unlinked at once, so no file outlives the factorization; no Mmap dependency).
* **GMRES-IR** (`BLAB_COUPLED_DENSE_GMRES_IR`): accepted only at the same Float64 backward-error test.

`dense_solver_diagnostics` reports `lu_float32_split_refined`, `lu_float32_split_gmres_ir` or
`lu_float64_fallback`, plus `dense_gmres_ir_iterations`, `dense_split_border_columns/rows`,
`dense_split_stray_entries` (expected 0) and `dense_spill`.

## Identities (all coupled backends)

The coupled caches used to hold the P1 × P1 and P1 × DP0 L2 identities as dense host matrices
(N_B² + N_B·F scalars per quadrature bundle; on CUDA also as dense device copies for the whole job).
They are now `SparseMatrixCSC` (`assemble_l2_identity_sparse`). The CPU operator path uses
sparse-identity methods of `burton_miller_neumann_lhs/matrices` that give the same entries bit for
bit; ROCm and the CUDA four-operator path materialize dense copies only where they are consumed.

## CUDA

| Stage | Before | Now |
|---|---|---|
| Job-long device identities | dense P1×P1 + P1×DP0 (8 N_B² + 8 N_B·F) | none (sparse COO added during assembly) |
| Combined assembly | A (8 N_B²) + C (8 N_B·F) | A + C on flux-carrying columns (`BLAB_CUDA_COUPLED_COMPACT_FLUX`) |
| Dense LU | ComplexF32 `lu!` on the device, no refinement | same LU, refined on the host to the Float64 backward error (`SplitRefinedLU` with a device factor; host copy of the matrix, spillable to a scratch file) |

`BLAB_COUPLED_DENSE_REFINEMENT=0` restores the unrefined CUDA solve. A refinement that stalls on
CUDA falls back to a host `ComplexF64` LU (GMRES-IR needs host factors and is not used there).
