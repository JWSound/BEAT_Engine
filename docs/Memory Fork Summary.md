# BEAT Engine: memory optimizations for large coupled FEM–BEM models

Base: `JWSound/BEAT_Engine` `main` @ `f25cd8e` (v0.8.0 + #106), condensed coupled solver
(Metal, CPU) and the CUDA coupled path. Notation: N_B = BEM P1 dofs, F ≈ 2·N_B = BEM faces,
F_active = faces that carry interface/motion/prescribed flux, n = dense system order
(≈ N_B + 2·transducers under flux elimination). Sizes are for complex Float32 (8 B) unless noted.

**Status: CPU paths tested** (condensed reference suite, coupled solver tests, new unit tests,
compiled CPU bundle workload). **Metal and CUDA kernels are not yet validated on hardware**;
`validate_fork.sh metal|cuda` runs the GPU gates and an old-vs-new comparison.

## 1. Compact C columns (Metal and CUDA)
- C (P1 × DP0) is only ever multiplied by flux maps that are nonzero on a few faces: the
  FEM–BEM interface, BEM transducer surfaces and prescribed-velocity faces.
- The combined assembly now keeps only those columns: N_B × F_active instead of N_B × F.
  Pair kernels skip inactive trial faces when writing C; singular write-back maps and the
  identity term are restricted to the active columns. A is unchanged.
- Bit-identical retained columns on Metal; equal up to atomic summation order on CUDA.
- Products with a flux that is nonzero outside the active set raise an error rather than
  silently dropping it.
- Switches: `BLAB_METAL_COUPLED_COMPACT_FLUX`, `BLAB_CUDA_COUPLED_COMPACT_FLUX` (`auto` | `0`).
- Example: hole-field horn quarter model, 78k BEM vertices: C from ~97 GB to ~4 GB.

## 2. No host copies of A and −C (Metal)
- The combined Metal path copied A (`copy(host.a)`) and negated C into a new array (`-host.c`)
  before building the dense system, and kept both until the frequency finished.
- Now A and −C (negated in place) stay on the device-backed host views. A is copied straight
  into the dense system and the device storage is released immediately afterwards. The BEM
  operators and prefetched copies are dropped before the factorization is allocated.
- Saves 8·N_B² + 8·N_B·F at the peak.

## 3. Sparse L2 identities (all coupled backends)
- The coupled caches held the P1 × P1 and P1 × DP0 identities as **dense** host matrices for
  the whole job: N_B² + N_B·F scalars per quadrature bundle. On CUDA, dense complex device
  copies were added on top.
- Example (78k BEM vertices): ~73 GB of host RAM, ~146 GB of VRAM on CUDA.
- They are now `SparseMatrixCSC` (`assemble_l2_identity_sparse`), a few MB in size.
- The CPU operator path uses sparse-identity methods of `burton_miller_neumann_lhs/matrices`
  that reproduce the same entries bit for bit. CUDA's combined assembly adds the identities
  from sparse COO arrays. Dense copies are made only for ROCm and the CUDA four-operator path.

## 4. Split-precision dense storage (`SplitDenseMatrix`, Metal and CPU)
- Refined solves stored the dense system twice: once in ComplexF64 (the residual) and once in
  ComplexF32 (the LU factor), 24·n² bytes in total.
- The dominant BEM block arrives in ComplexF32, so a ComplexF64 copy of it adds no information.
  Only the rows and columns written from Float64 FEM data (Schur couplings, interface and
  transducer blocks) need it.
- New storage: ComplexF32 for every entry, plus the exact ComplexF64 remainder on a declared
  border of rows and columns. Writes accumulate in Float64 exactly as before, so the store
  equals `ComplexF32.(M)` and the reconstruction equals `M` bit for bit.
- Same LU factor, same refinement acceptance test. Only the residual's summation order differs.
- 24·n² → 16·n² (+ 16·n·border). Switch: `BLAB_COUPLED_DENSE_SPLIT` (`auto` | `0`).

## 5. Spilled refinement store
- The residual copy of the ComplexF32 matrix can live in an unlinked scratch file that is
  streamed by column panels (no Mmap dependency). `hi` is then factored in place.
- 16·n² → 8·n² resident, plus 8·n² on disk, read 1–3 times per frequency.
- Switches: `BLAB_COUPLED_DENSE_SPILL` (`auto` | `on` | `off` | directory); `auto` spills when
  16·n² exceeds `BLAB_COUPLED_DENSE_SPILL_FRACTION` (default 0.5) of RAM.

## 6. GMRES-IR instead of the Float64 LU fallback
- A stalled refinement used to fall back to a full ComplexF64 LU: +16·n², often fatal on large
  models.
- It now continues as restarted GMRES-IR (Carson–Higham): ComplexF64 Arnoldi, with the
  ComplexF32 LU factors applied in Float64 arithmetic as preconditioner (blocked, one streamed
  pass per application). All right-hand sides run in lockstep.
- Accepted only at the same Float64 backward-error test. Converges where κ·eps(Float32) ≫ 1.
  The Float64 LU remains as the last resort and records why it ran.
- Switch: `BLAB_COUPLED_DENSE_GMRES_IR` (`1` | `0`).

## 7. Refinement on CUDA
- The CUDA coupled path factored the dense system in ComplexF32 on the GPU with **no
  refinement**. According to the engine's own notes, plain Float32 LU missed references by
  0.03–8.9 dB on example models.
- The F32 LU stays on the device. It is now wrapped in the same refinement against the host
  copy of the matrix (spillable). The diagnostics report `dense_solver` as on Metal.
- `BLAB_COUPLED_DENSE_REFINEMENT=0` restores the old solve.

## Combined effect (peak memory, flux elimination, n ≈ N_B)

| Model (N_B) | Before (Metal) | After (Metal, no spill / spill) | Before (CUDA VRAM) | After (CUDA VRAM) |
|---|---|---|---|---|
| 23k | ~35 GB | ~10–17 / ~5–9 GB | ~26 GB | ~5–9 GB |
| 55k | ~180 GB | ~50–65 / ~30–35 GB | ~145 GB | ~25–50 GB |
| 78k | ~360 GB | ~100–125 / ~55–65 GB | ~290 GB | ~50–100 GB |

Estimates read off the code paths: they include the dense identities, assume one quadrature
bundle, and are not measured yet. The CUDA range depends on whether A and the dense system
coexist on the device while the system is built.

## Diagnostics
`dense_solver` reports `lu_float32_split_refined`, `lu_float32_split_gmres_ir` or
`lu_float64_fallback`, together with `dense_refinement_iterations`,
`dense_refinement_backward_error`, `dense_gmres_ir_iterations`,
`dense_split_border_columns/rows`, `dense_split_stray_entries` (expected 0) and `dense_spill`.

## Not covered yet
- Dense LU through Apple Accelerate (AMX/SME) instead of OpenBLAS; benchmark pending.
- The exterior-only solvers still build dense identities; the same fix applies there.
- The Deploy path still uses the four-operator Metal assembly.

## Optional: Accelerate dense LU (`BLAB_DENSE_LU_LIBRARY`)

- Host dense LU factorizations can use Apple Accelerate's ILP64 `getrf` instead of OpenBLAS:
  `BLAB_DENSE_LU_LIBRARY=accelerate`, or `auto` (Accelerate only where the chip has SME, M4 and later).
- Default `blas`: on an M1 Max (no SME) Accelerate's `cgetrf` was slower than OpenBLAS
  (340-500 vs 520-570 GFLOP/s, n = 4096-24576). M4/M5 owners: run `tools/bench_dense_lu.jl` to decide.
- Only the factorization changes library; the factors are an ordinary `LU`, solves are unchanged.
