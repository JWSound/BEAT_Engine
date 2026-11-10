# Coupled dense memory on Metal (condensed solver)

Per frequency the condensed coupled solve used to hold, at its peak (N_B BEM vertices,
F ≈ 2 N_B BEM faces, n ≈ N_B + 2·transducers the dense order under flux elimination):

| Stage | Before | Now (defaults) |
|---|---|---|
| Combined BEM assembly (A, C) | A (8 N_B²) + C (8 N_B·F) on shared storage | A + C on the flux-carrying columns only (8 N_B·F_active) |
| Host combine | + host copy of A (8 N_B²) + negated C copy (8 N_B·F) | no copies: A and −C stay on the device-backed host views until consumed |
| Dense system + factor | ComplexF64 matrix (16 n²) + ComplexF32 factor (8 n²); F64 LU fallback +16 n² | ComplexF32 store + border corrections (8 n² + 16 n·w) + factor (8 n²); with spill the store is a memory-mapped file (8 n² resident) |
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
* **Spill** (`BLAB_COUPLED_DENSE_SPILL`): same arithmetic, the residual store is read from a mapped file
  (unlinked at once, so no file outlives the factorization).
* **GMRES-IR** (`BLAB_COUPLED_DENSE_GMRES_IR`): accepted only at the same Float64 backward-error test.

`dense_solver_diagnostics` reports `lu_float32_split_refined`, `lu_float32_split_gmres_ir` or
`lu_float64_fallback`, plus `dense_gmres_ir_iterations`, `dense_split_border_columns/rows`,
`dense_split_stray_entries` (expected 0) and `dense_spill`.
