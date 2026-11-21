# Fast-multipole BEM for the coupled solver — plan and status

Goal: solve the coupled FEM–BEM systems of real loudspeaker models (horns with FEM
prechambers, thermoviscous walls, electrodynamic transducers, symmetry planes) at sizes
where the dense LU no longer fits (≥ 150k BEM unknowns), locally on a Mac and on rented
GPUs, with results that agree with the dense solver to a stated tolerance.

Accuracy rule: the FMM path reproduces the *dense matrix the engine already builds*
(same regular rule, same Duffy rules, same image deltas), so every comparison is
apples to apples; the only new error sources are the FMM tolerance and the iterative
solver tolerance, both reported per solve.

## Building blocks

* FMM3D (Flatiron Institute, Apache-2.0), via `FMM3D.jl` / `FMM3D_jll`: Helmholtz charges +
  dipoles, multiple densities per call, potential and gradient at targets. Prebuilt for
  aarch64-apple-darwin, x86_64-linux, Windows. CPU (OpenMP). Kernel convention checked at
  run time (`fmm_kernel_factors`): FMM3D's Helmholtz kernel is exp(ikr)/(4πr).
* Galerkin Burton–Miller as in `BeatEngineCpuBurtonMiller.jl`:
  `A = 0.5 M_pp − D + (i/k) H` (hypersingular in Maue form), `C = S + (i/k)(K′ + 0.5 M_pq)`.
  Far field of A: one dipole density + six charge densities (normal and surface-curl
  vectors). Far field of C: one charge density per drive, potential + gradient.
* Near field: sparse correction `Duffy(adjacent) − regular(adjacent) + image deltas`,
  built with the engine's own pair kernels (`_beat_cpu_bm_pair_blocks`,
  `_beat_cpu_bm_regular_pair_blocks`), so it is bit-compatible with the dense build.

## Milestones

1. [done] Exterior operator `A·x` and `C·q` matrix-free, all symmetry modes (`off`, `x`,
   `xy`), equal to the dense CPU assembly to < 1e-11 relative (tests/fmm_operator_tests.jl).
2. [in progress] GMRES with a near-field sparse-LU preconditioner; iteration counts on
   spheres (3–6 iterations up to ka = 10) and on the rigid HF piece (tests/fmm_solve_probe.jl).
   HF piece (5 mm mesh, 5.7k nodes, xy symmetry): product matches dense to 1e-14 (A) and
   5e-8 (C) at 10 and 20 kHz. Cost on 2 cores: 55–80 s per A-product vs 124 s for the whole
   dense assembly. FMM3D.jl drops the library's error flag; with 7 densities the plane-wave
   workspace did not fit in 7 GB and the wrapper silently returned wrong potentials
   (135 % error). Fixed by `fmm_hfmm3d_checked` (own ccall, `Ref` error flag) with automatic
   per-density fallback.

## Measurements so far (container: 2 cores, 7 GB)

HF piece, 5 mm mesh (5.7k nodes, 11.2k faces, xy symmetry -> 134k FMM sources), 10 kHz:

| | value |
|---|---|
| A-product vs dense | 6e-15 (3-point rule everywhere) |
| C-product vs dense | 5e-8 |
| one A-product, eps 1e-4 | 32 s @ 10 kHz, 55 s @ 20 kHz (7 single-density FMM calls) |
| FMM3D peak memory | 3.3 GB @ 10 kHz, 5.2 GB @ 20 kHz (grows with frequency) |
| GMRES, near-field LU preconditioner, rtol 1e-9 | 47 iterations, solution vs dense LU 6e-10 |
| GMRES, Jacobi | 87 iterations |
| 3.43 mm mesh (11k nodes) | out of memory in 7 GB |

Findings:
* Krylov.jl's default `atol = sqrt(eps)` stopped GMRES early on small right-hand sides
  (1-2 % solution error with "solved = true"); every solve must pass `atol = 0`.
* The near preconditioner must include image (symmetry) pairs: without them 157 iterations,
  with them 14 to 1e-6 / 47 to 1e-9.
* A one-point far-field rule is not usable: 1-2 % product error on a sphere at a near radius
  of 2-4 h. The far field stays on the dense build's 3-point rule.
* FMM3D's results at eps = 1e-4 match the dense product to 1e-15 here, and the time is close to
  a direct sum: at this size and kD the library is effectively not compressing. Whether it
  does at 10-100x more sources, and how its memory grows, decides whether FMM3D is the right
  far-field engine. That measurement needs the Mac (and a bigger memory than 7 GB).

## FMM3D on its own (random surface points, 0.35 x 0.35 x 0.2 m, 10 kHz, eps 1e-4, 2 cores)

| sources | targets | time | peak memory | direct sum (est.) |
|---|---|---|---|---|
| 67k | 17k | 2.2 s | 1.0 GB | 0.7 s |
| 134k | 34k | 3.9 s | 1.4 GB | 3.0 s |
| 268k | 67k | 7.9 s | 3.2 GB | 12 s |

* It does compress: time doubles with the point count, the direct sum quadruples. At the
  piece size the two are still about equal, which is why the piece product looked "direct".
* Seven densities in one call: 1.2 s per density instead of 2.3 s (about 2x), but 5x memory.
* eps hardly changes the time (1e-3 .. 1e-6 within 25 %): the near-field direct part dominates.
* `hfmm3d_ndiv` from FMM3D.jl returned wrong potentials (relative error 1.0) for ndiv 10-40,
  silently, like the error flag issue: the leaf size must be tuned through our own checked
  ccall, never through the wrapper.
* Memory ~12 kB per source at 10 kHz and growing with frequency: the binding constraint for
  the big models on a 32 GB Mac.

Extrapolation (to be confirmed on the Mac): two-horn reduced model at lambda/10 (~180k nodes,
~4M sources with images) ~ 2 min per density on 2 cores -> ~15-20 min per product on 2 cores,
~3-5 min on 10 cores; x 30-50 GMRES iterations -> 1.5-4 h per frequency. Too slow for sweeps
as it stands; the levers are leaf-size tuning, combined densities, Float32 near field and fewer
images, and if those do not give ~10x, a different far-field engine.

## Coupled prototype (tests/fmm_coupled_proto.jl, frozen engine fixture: 1214 BEM vertices,
## 842 FEM vertices, 106 interface DOFs; layout :none, no transducers)

* Matrix-free product [Schur p_G - L q; A p_B + C B q; T_G p_G - T_B p_B] equals the engine's dense
  condensed matrix to 1e-15 in every block row (sign: bem_interface_block = +C * bem_flux).
* GMRES with the near-field block preconditioner (same block system, A and C replaced by their
  exact near-pattern values, Schur and traces exact; sparse LU): 12 / 13 / 16 iterations at
  500 / 2000 / 5000 Hz to rtol 1e-9. Unpreconditioned: ~240.
* The coupled matrix has cond ~1e9 and block row norms Γ 0.12, BEM 0.005, continuity 14.6: an
  unscaled residual test leaves the BEM rows at 2e-6 (solution error 5e-4 vs dense LU). With the
  rows scaled by the preconditioner's row norms: 21 iterations, solution error 1.5e-7.
  Every coupled solve must be row-equilibrated.
* Scalable preconditioner done (`fmm_coupled_preconditioner`): block elimination with one sparse
  LU of the BEM near block, a dense LU of the Schur block and a dense |Q| x |Q| interface system,
  the A_n^-1 R_n F term formed in column blocks. Equals the sparse LU of the whole near system
  to 1.6e-14 on the fixture; same iteration counts.
* Transducers (tests/fmm_coupled_transducer_proto.jl, fixture + one electrodynamic transducer on the
  Radiator group, 1218 BEM faces in motion, transducer condensation off): only the BEM row blocks
  [B,B], [B,Q], [B,mech] need the FMM (A, and C applied to flux + motion data); the product equals
  the dense matrix to 4e-14 / 1e-13 at 500 / 3000 Hz. Every other block is reused as built.
* Generic bordered preconditioner (`fmm_bordered_preconditioner`): sparse LU of A_n, dense Schur
  complement on all non-BEM unknowns (any layout, any transducer count). With row scaling:
  19 / 24 GMRES iterations at 500 / 3000 Hz, solution vs dense LU 1e-6 / 4e-6 at rtol 1e-9.
  Its dense block costs 16 |S2|^2 bytes: fine to |S2| ~ 16k (4 GB); beyond that use the
  Schur-of-Schur form of `fmm_coupled_preconditioner`.
* Open: interface-elimination layouts (:flux), engine integration.
* Mac benchmark: kit `20_fmm_scale.sh` (piece 5 / 3.43 / 2.45 / 1.7 mm, 10 and 20 kHz, single vs
  combined densities, /usr/bin/time -l for peak memory).

* `FmmCoupledSolve` (drop-in for `system.factorization`: `size` and `\` over right-hand-side
  columns, row-equilibrated GMRES with the bordered preconditioner): on the transducer fixture at
  1 kHz, 2 columns, 19 iterations each, row-scaled residual 8e-10, solution vs dense LU 9.5e-6
  (the coupled matrix's condition number ~1e9 turns rtol 1e-9 into ~1e-5).

* Profile of one A-product (piece 5 mm, 10 kHz, eps 1e-4, 2 cores): one charge FMM call 4.7 s,
  one dipole call 5.3 s, 6 + 1 calls = 33 s = the whole product; the Julia-side packing,
  projection and sparse correction are negligible. Speed therefore comes only from fewer/faster
  FMM calls: combined densities (measured 2x per density, 5x memory), more cores, or a different
  far-field engine. A coupled GMRES iteration also needs one C-product (one more call).

## Mac M1 Max measurement (8 performance cores), 2026-11-21 — FMM3D ruled out

| piece mesh | nodes | f | A-product, single densities | memory | combined | memory |
|---|---|---|---|---|---|---|
| 5 mm | 5.7k | 10 kHz | 14.6 s | 3.3 GB | 5.5 s | 11.8 GB |
| 5 mm | 5.7k | 20 kHz | 23.0 s | 6.3 GB | 10.1 s | 14.6 GB |
| 3.43 mm | 11.1k | 10 kHz | 20.4 s | 8.9 GB | 9.1 s | 14.8 GB |
| 3.43 mm | 11.1k | 20 kHz | – | – | 16.7 s | 14.7 GB |
| 2.45 mm | 21.7k | 10 kHz | 43.1 s | 15.4 GB | – | – |

8 cores give only ~2.2x over 2 cores; memory is 6-30x the *dense* matrix at these sizes
(dense 5.7k = 0.5 GB). Extrapolated to 180k nodes: minutes per product and >100 GB. FMM3D is
not a viable far-field engine for this problem class.

## H-matrix (ACA) compressibility, 20 kHz, tol 1e-3, eta = 1 (one point per face proxy, no images)

| piece mesh | faces | H storage / dense | max rank | complex64: dense -> H |
|---|---|---|---|---|
| 5 mm | 11.2k | 0.32 | 160 | 0.9 -> 0.30 GB |
| 3.43 mm | 22.0k | 0.22 | 354 | 3.6 -> 0.78 GB |
| 2.45 mm | 43.0k | 0.14 | 459 | 13.8 -> 1.9 GB |
| 1.7 mm | 87.3k | 0.091 | 1403 | 56.7 -> 5.2 GB |

Storage grows ~N^1.4 (x2.6-2.7 per doubling). Extrapolated: two-horn reduced-height model at
lambda/10 (~350k faces / ~180k P1 unknowns): ~12-20 GB complex64 vs ~260 GB dense; half of that
in FP16. Open: images (symmetry) raise ranks near the planes; the Galerkin hypersingular blocks
vs the point-kernel proxy; the coarse-level max ranks grow fast with frequency.

## Low-precision dense storage (piece 5 mm, row-scaled, exterior BM)

| storage | error, no refinement | GMRES-refined against full-precision product |
|---|---|---|
| Float16 | 6e-4 (20 kHz), 2e-4 (10 kHz) | 2 iterations, 7e-7 / 3e-7 |
| BFloat16 | 5e-3 / 3e-3 | 3 iterations, 8e-7 / 3e-7 |

## H-matrix with exact entries (2026-11-21, container, 2 threads)

Code: /home/claude/hmat (kit copy: kit/hmat). `FastOracle` gives exact Galerkin BM entries
(regular 3-point rule with images + sparse singular/near correction + row weights + 0.5 M);
checked against the dense engine matrix to 1e-15. Cluster tree on P1 vertices (leaf 48),
eta = 1 admissibility tested against the original AND every image cluster; ACA (tol/10)
+ QR/SVD recompression to tol. Threaded build (`hbuild_threaded`), index maps instead of Dicts,
single-row/column fast paths. RHS by direct sum over driver sources (`rhs_direct`, matches FMM to 5e-8;
FMM3D itself OOMs at 20 kHz). Preconditioner: block-Jacobi from diagonal leaf blocks
(near-field LU OOMs at 11k/20 kHz in 7 GB).

| mesh | n | f | tol | storage vs dense | GB (c64) | max rank | build | GMRES its | solve | peak RSS | sol. err vs dense |
|---|---|---|---|---|---|---|---|---|---|---|---|
| piece5 | 5690 | 10 kHz | 1e-3 | 0.394 | 0.10 | 18 | 105 s | 34 (near LU) / 42 (block-Jacobi) | 3 s | – | 3.9e-4 |
| piece_h343 | 11108 | 20 kHz | 1e-3 | 0.285 | 0.26 | 27 | 401 s | 136 (block-Jacobi) | 20 s | 5.0 GB | – |
| piece_h245 | 21675 | 20 kHz | 1e-3 | OOM in the 7 GB container during operator setup → run on the Mac | | | | | | | |


Low-precision H-matrix storage (piece5, 10 kHz, tol 1e-3; `hconvert_scaled`, per-ROW scale kept in Float32):

| storage | operator rel. err | GMRES its | diff vs c128 H solution | + refinement (inner its) | vs dense |
|---|---|---|---|---|---|
| FP16 | 7.1e-5 | 42 | 2.0e-4 | 7.4e-7 (63) | 3.9e-4 |
| BF16 | 5.9e-4 | 42 | 1.6e-3 | 7.4e-7 (69) | 3.9e-4 |

Per-BLOCK scaling with FP16 fails completely (GMRES diverges, although the operator error is 1e-4):
rows of a block differ by many orders of magnitude and FP16 subnormals wipe out the small rows.
Per-row scaling is mandatory for FP16; BF16 tolerates either. Since the ACA tolerance (1e-3) already
dominates, FP16 storage needs no refinement: half of complex64 memory (4 bytes per entry).
Mac run script: kit/30_hmat_mac.sh (11k / 22k / 43k meshes at 20 kHz).


Preconditioner (piece_h343, 11108 nodes, 20 kHz, saved H): block-Jacobi on cluster-tree nodes of size <= S,
each block assembled from ALL H blocks (dense + low-rank) inside the cluster (`cluster_block_precond`):

| S | blocks | memory (c128) | GMRES its | solve |
|---|---|---|---|---|
| leaf diag only (old) | 334 | – | 136 | 22 s |
| 48 | 335 | 0.01 GB | 98 | 9 s |
| 200 | 80 | 0.02 GB | 85 | 6 s |
| 500 | 34 | 0.06 GB | 75 | 6 s |
| 1000 | 15 | 0.13 GB | 34 | 3 s |
| 2000 | 8 | 0.26 GB | 24 | 6 s |
| 4000 | 4 | 0.46 GB | 19 | 3 s |

Default S = 1000 (memory n*S*16 B: ~5 GB c128 / 2.4 GB c64 for n = 300k). piece5 10 kHz: 23 its.
FP16 (row-scaled) on piece_h343 20 kHz: operator 7.8e-5, solution 4.7e-4 vs c128 H, same iterations.


Mac M1 Max run (8 threads, 2026-11-21, kit/30_hmat_mac.sh, cluster block-Jacobi S = 1000, 20 kHz, tol 1e-3):

| mesh | n | storage vs dense | GB c64 | max rank | build | GMRES its | solve | FP16 diff | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| piece_h343 | 11108 | 0.285 | 0.26 | 27 | 256 s | 34 | 3 s | 4.7e-4 | 6.2 GB |
| piece_h245 | 21675 | 0.164 | 0.57 | 27 | 576 s | 76 | 11 s | 3.6e-4 | 16.7 GB |
| piece_h17 | 43856 | 0.090 | 1.29 | 28 | 1240 s | 86 | 28 s | (op 7.6e-5) | 21.4 GB |

Extrapolation to n = 180k (dense 260 GB c64): storage fraction ~0.027 -> ~7 GB c64 / ~3.5 GB FP16,
+ preconditioner ~1.4 GB; GMRES ~100-150 its, ~2-3 min; CPU build ~1.5 h per frequency on the M1 Max
(build time ~n^1.1) -> GPU build is the critical item.

Max rank stays at 27 and storage fraction falls ~n^-0.8: H storage grows ~n^1.2. Peak RSS is
build garbage (per-call allocations) and must be cut before larger meshes. Build only 1.6x faster
on 8 threads than on 2 container threads: threading efficiency is poor (allocation/GC bound).


Dense reference on the Mac (HMAT_DENSE=1, 20 kHz, tol 1e-3, kit/31_hmat_dense_mac.sh), greedy dynamic build:

| mesh | n | build | dense assembly | LU+solve | H c128 err vs dense | H FP16 err vs dense |
|---|---|---|---|---|---|---|
| piece_h343 | 11108 | 216 s | 146 s | 13 s | 7.9e-4 | 9.6e-4 |
| piece_h245 | 21675 | 423 s | 616 s | 101 s | 7.4e-4 | 8.0e-4 |

GPU kernel (KernelAbstractions, hmat_ka.jl, entry-wise regular BM integrand with images), Metal Float32 on the M1 Max
(kit/32_hmat_gpu_mac.sh): piece_h343 20 kHz, all 14.0 M dense near-field entries in 3.0 s (214 ns/entry),
worst rel err vs Float64 CPU oracle (regular+correction) 8.1e-6; piece_h245: 24.9 M entries in 5.3 s (211 ns/entry), worst 1.5e-5. CPU check (container): Float64 identical to reg_block!
to 1e-15; Float32 2.7e-6 worst.


Hybrid build on the Mac (kit/33_hmat_hybrid_mac.sh: dense near-field blocks on Metal Float32 with exact Float64 near
entries, ACA on 8 CPU threads), piece_h343 20 kHz: device 6.2 s, ACA 193 s, total 202 s (CPU-only 216 s);
solution err vs dense 7.9e-4, FP16 9.6e-4 (identical to CPU-only). => the ACA rows/columns are the real
bottleneck (~95 % of build); next: batched ACA row/column evaluation on the GPU.
piece_h245: device 9.6 s, ACA 392 s, total 404 s (CPU-only 423 s), GMRES 76.
Batched device ACA (aca_batched / hbuild_gpu, all active blocks per round in one launch, 43 rounds on piece5):
validated on the CPU Float32 backend, piece5 10 kHz: err vs dense 3.9e-4 (identical). Mac test: kit/34_hmat_gpufull_mac.sh.


Full device build on the Mac (kit/34_hmat_gpufull_mac.sh, Metal Float32 + exact Float64 near entries, batched ACA):

| mesh | n | near exact | dense | ACA (rounds, device) | recompress | total build | CPU-only | hybrid | err vs dense |
|---|---|---|---|---|---|---|---|---|---|
| piece_h343 | 11108 | 3.1 s | 3.0 s | 18.2 s (74, 11.2 s) | 10.3 s | 37 s | 216 s | 202 s | 7.9e-4 (FP16 9.6e-4) |
| piece_h245 | 21675 | 4.2 s | 5.3 s | 43.0 s (69, 26.6 s) | 16.5 s | 72 s | 423 s | 404 s | – |
| piece_h17 | 43856 | 6.6 s | 10.9 s | 96.3 s (67, 61.2 s) | 38.8 s | 157 s | 1240 s | – | – (GMRES 86, 28 s; FP16 3.4e-4 vs H) |

Next bottlenecks: CPU recompression (QR/SVD per block) and host-side ACA bookkeeping/transfer.
After BLAS=1 inside the threaded recompression + skipping the exact near fixup for admissible blocks without near
entries (4 of 3346 need it on piece5): h343 recompress 10.3 -> 2.3 s, build 37 -> 30 s; h245 16.5 -> 5.1 s, build 72 -> 63 s.
Remaining: host side of ACA rounds (h343 18.9 s total vs 10.8 s on the device).

Notes: build time is dominated by the dense near-field leaf blocks (≈1.3e9 kernel evaluations at 5.7k);
it is embarrassingly parallel and a natural GPU kernel. Peak RSS is mostly allocation garbage
during the build (heap hint 3 GB); the H-matrix itself is 0.5 GB c128.

## Engine integration design (not implemented; engine files untouched)

Switch: `BLAB_COUPLED_BEM_SOLVER=fmm` (default `dense`), CPU and Metal backends. FMM tolerance
`BLAB_FMM_EPS` (default 1e-6), GMRES `BLAB_FMM_RTOL` (default 1e-8), `BLAB_FMM_MAXITER`.

Where it plugs in (src/BeatEngineCoupledCondensed.jl):

1. `build_condensed_coupled_system` (~2086). Under fmm:
   * force `interface_elimination = :none` (the :flux/:pressure layouts fold dense
     `bem_interface_block` products into the system; with an operator there is nothing to fold),
     and `transducer_condensation` keeps its current choice (both layouts verified in the
     prototypes);
   * skip `_assemble_condensed_bem_operators` / `_combine_condensed_bem_operators!` (no `bem_lhs`,
     no `bem_rhs_operator`); build `FmmBurtonMillerOperator` once per frequency from the same
     `prepared` quadrature bundle, singular order and symmetry mode;
   * build every non-BEM block exactly as now (Schur from `condensation`, `fem_load`,
     `fem_trace`, `bem_trace`, mechanical/electrical blocks) into a sparse "rest" matrix, with the
     three BEM row blocks left empty;
   * `bem_prescribed_rhs = -fmm_rhs(op, bem_prescribed_neumann)` (the dense code forms
     `bem_rhs_operator * bem_prescribed_neumann`, and `bem_rhs_operator = -C`, so the sign must
     follow the prototype's verified convention);
   * `bem_motion_block` is never formed: the motion flux enters the operator product;
   * `factorization = FmmCoupledSolve(...)`: a struct with `size(::, 1)` and `\` (matrix of right-hand
     sides) that runs row-equilibrated, right-preconditioned GMRES per column with the bordered
     near-field preconditioner. This is the same contract `SplitGMRES` already satisfies, so
     `solve_condensed_coupled_excitations` (line ~2938, `system.factorization \ rhs`) and
     `_solution_from_parts` need no change.
2. `retain_interface_radiation` replays `lu(bem_lhs) \ (interface_block * flux)`: under fmm the
   replay becomes a GMRES on the exterior operator alone with the near preconditioner (already
   prototyped in tests/fmm_solve_probe.jl).
3. Field outputs (polars) evaluate the boundary solution at observation points and do not use the
   dense matrix: unchanged.
4. Wall impedance / thermoviscous boundaries: check whether any term lands in `bem_lhs`; if so it is
   a sparse (mass-like) addition and goes into the operator's near/identity part.

Diagnostics added per frequency: `bem_solver = "fmm"`, `fmm_eps`, `fmm_sources`,
`fmm_product_s` (mean), `fmm_setup_s`, `gmres_iterations` (per excitation), `gmres_relative_residual`
(row-scaled and unscaled per block), `preconditioner_setup_s`, `preconditioner_dense_order`,
peak memory.

Reference gate before merge: the frozen coupled fixture (with and without transducer, with
symmetry) at 3+ frequencies must match the dense condensed solution to the GMRES tolerance times a
fixed factor, solution and field outputs; plus the archived HF-piece references (3.43 / 2.45 /
1.7 mm runs, 20261120-conv) on the Mac.

## Next

* Mac: scaling of one A-product (time, memory) on 5 mm, 3.43 mm, 2.45 mm, 1.7 mm piece meshes at
  10 and 20 kHz; GMRES iterations on 3.43 mm.
* Meanwhile (library-independent): milestone 3, the coupled matrix-free operator and block
  preconditioner, tested on the engine's coupled fixtures against the dense condensed solver.
* If FMM3D does not compress at these kD: evaluate HMatrices.jl (ACA, Julia) and a directional
  / wideband FMM before committing.

## Independent review (ChatGPT architecture note + diagnostics, 2026-11-21; kit/review/)
Confirms: block sampler from local element integrals, ACA H-matrix as first path, scaled FP16 factors,
near/singular blocks must not be single precision (found independently: Float32 regular part cancels
by up to 11 orders against the correction on symmetry-plane image pairs -> near entries now exact Float64).
Adopt:
1. Acceptance metric = max complex relative error of field pressure over frequencies/points/excitations
   (compare_complex.py), not L2 of the boundary solution. Needs BLAB-result -> npz adapter (shares code with VACS export).
2. Compress C as well; apply large FEM Schur complements implicitly; measure interface DOFs first.
3. 20 kHz rank survey on the real large model (two-horn) before committing; directional H2 as fallback.
4. Band meshes per frequency band for sweeps.
5. Symmetry also requires symmetric excitation (two-horn drive phase).
Disagree (partly): an uncompressed FP64 reference operator is not available on large models; use a tighter-tolerance H as reference.

## Coupled-system sizes (archived BLAB projects, node counts from the .msh files; interface = BEM surface nodes shared with FEM meshes)

| project | BEM surface nodes | FEM nodes | interface nodes | dense Schur (c128) | dense C = n_B x n_q(≈2 x interface faces) c128 |
|---|---|---|---|---|---|
| quarter horn h343 (86.6k order) | 78036 | 60203 | 3928 | 0.25 GB | ~9.8 GB |
| HF piece h343 | 11108 | 23456 | 4306 | 0.30 GB | ~1.5 GB |
| HF piece h170 | 43856 | 146845 | 16804 | 4.5 GB | ~23.6 GB |

=> the FEM Schur complement stays affordable dense up to ~15-20k interface nodes; the BEM operator C (P1 rows x interface
DP0 columns) is the next memory item to compress (same H-matrix machinery, rectangular, face cluster tree for columns).

## H-matrix of the right-hand operator -C (hmat_c.jl, hmat_c_test.jl)
Entry kernel _negc_entries! (P1 row x face column, images, -G - eta dG/dn_x) matches rhs_direct to 1.1e-15 (Float64) /
1.6e-7 (Float32) on piece5. Exact Float64 near entries (reg + rhs_correction), row weights, -0.5 eta Ipd as for A.
Rectangular H-matrix: P1 vertex tree x face-centroid tree (leaf 48), admissibility with face boxes padded by the face size,
original + image clusters. piece5 10 kHz tol 1e-3 (CPU Float32): storage 0.374 of dense, max rank 16, driver RHS vs direct 3.2e-4.
Mac job: kit/35_hmat_c_mac.sh (11k/22k/43k at 20 kHz).

Mac Metal, 20 kHz, tol 1e-3 (kit/35_hmat_c_mac.sh):

| mesh | n x faces | C storage vs dense | GB c64 (dense) | max rank | build (near/dense/ACA[device]/recompress) | driver RHS err vs direct |
|---|---|---|---|---|---|---|
| piece_h343 | 11108 x 21989 | 0.262 | 0.48 (1.82) | 26 | 17 s (1.8/1.6/9.4[2.1]/2.4) | 2.6e-4 |
| piece_h245 | 21675 x 43032 | 0.146 | 1.01 (6.95) | 26 | 34 s (2.0/2.8/21.2[4.7]/4.9) | 3.1e-4 |
| piece_h17 | 43856 x 87260 | 0.079 | 2.25 (28.5) | 26 | 63 s (2.8/6.0/41.6[9.2]/7.3) | 4.8e-4 |

The random-phase flux check (1e-11 vs FMM) is not meaningful: incoherent far contributions cancel and the near part
dominates. Use smooth excitations (driver, plane-wave phase) for this check. ACA host bookkeeping dominates the C build
(device 9 s of 42 s on h17) -> next optimisation target. gpu-full A on h17 with the BLAS fix: build 157 -> 132 s.

## Coupled prototype with H(A) and H(-C) (hmat_coupled_proto.jl, engine fixture femvolume + exterior_conforming + 1 transducer)
2 kHz, tol 1e-4, CPU Float64: n 1485 (Γ 163, BEM 1214, q 106, mech 1, elec 1). Coupled product vs dense: BEM rows 3.6e-7,
other blocks exact. GMRES with the existing bordered near preconditioner + row scaling: 24 its; solution vs dense LU:
all 6.7e-6, BEM p 1.6e-6, q 7.4e-6, mech 2.0e-8, elec 5.4e-10 (unit voltage drive).

## Real BLAB project through the H path (HF piece h343, 20 kHz, 6 transducers x 20 excitations, symmetry xy)
Development hook BLAB_HMAT_DUMP (patch kit/patches/hmat_dump_hook.diff, applied only for the run and reverted by
kit/36_hmat_real_model_mac.sh): with layout :none and BLAB_COUPLED_DENSE_SPLIT=off the condensed solver writes the
non-BEM part of the coupled matrix (sparse), interface/motion flux, BEM mesh, identities, 64 dense A and interface rows,
rhs and the dense solution as raw .bin files (no extra package dependencies: the compiled Metal bundle cannot load
Serialization). Checker: hmat/hmat_dump_check.jl.
System: n 19656 = Γ 4254 + BEM 11108 + q 4254 + mech 20 + elec 20; bem_motion_flux empty (transducer condensation on).
Finding: the driver's system is the complex conjugate of the oracle's (engine rows = conj(oracle) to 3.8e-6, the
Float32 Metal assembly level); the checker now detects this and solves the conjugate system.
Result (Metal, 20 excitations, kit/37): engine A rows = conj(oracle) 3.8e-6; interface rows vs engine 3.6e-4 (tol 1e-3) /
3.5e-5 (tol 1e-4); H(A) 0.284/0.362 of dense, H(C) 8184 cols 0.251/0.305; build A 30/39 s, C 8/10 s; GMRES 59-62 its,
14-16 s per excitation. Max errors vs the engine's dense solution, over the 20 excitations: BEM p 1.2e-3..2.0e-2,
q 9e-3..1.6e-1, mech 7e-5..3e-3, elec 1e-7..4e-6 -- IDENTICAL for tol 1e-3 and 1e-4, so not compression error.
Hypothesis: conditioning (cond ~1e9) amplifies the reference's own Float32 assembly error (3.8e-6 per entry), or a block
of the coupled operator is still mismatched. Next: scaled residuals of both solutions against the H operator (kit/37),
and a Float64 dense reference (beat_cpu run was also float32: precision comes from solver_options).
Diagnosis: the two engine dense references disagree with EACH OTHER as much as H disagrees with them.
Metal float32 vs CPU float32 dense solutions (same project, same frequency), max |diff|/max|ref|:
exc 1: BEM p 1.0e-3, q 1.2e-2, mech 2.2e-4, elec 1.5e-7; exc 7: 1.2e-2, 6.9e-2, 1.9e-3, 2.4e-6; exc 17: 1.4e-2, 1.6e-1, 1.7e-3, 2.2e-6.
H vs CPU float32: exc 1: 1.1e-3, 7.0e-3, 1.2e-4, 4.7e-8 (same size). The dense solutions satisfy their own non-BEM rows
to 1e-5..1e-7 (Float32 level); the interface flux (|q| ~ 1e4 vs |p| ~ 5) nearly cancels in the BEM rows, so per-entry
Float32 assembly errors (~4e-6) are amplified to percent level in q and p. The H solution (exact Float64 near entries,
tol 1e-3/1e-4, same result) is at least as close to the truth as either Float32 reference; a Float64 reference run
(kit/39, solver_options.precision=float64 on beat_cpu) decides.

### Float64 reference (kit/39: request solver_options.precision=float64 on beat_cpu, dense_solver lu_float64, 350 s, +17 GB, +5.5 GB swap)
H (Metal, tol 1e-4) vs Float64 dense, worst over 20 excitations: BEM p 1.6e-4, q 2.9e-4, mech 3.8e-6, elec 5.1e-9
(rel L2 all <= 4.8e-4). => the H path meets the accuracy target on the real coupled model at 20 kHz.
Float32 engine runs vs Float64 (boundary solution), worst over 20 excitations:
Metal f32: BEM p 2.0e-2, q 1.6e-1, mech 3.1e-3, elec 3.9e-6; CPU f32: 1.3e-2, 1.2e-1, 2.8e-3, 3.4e-6.
Field outputs (polar exterior pressure, 20 excitations x 73 angles): Float32 vs Float64 max complex error 3e-3 of the
array maximum; per driver within 6 dB of its maximum <= 0.04 dB, within 20 dB <= 0.16 dB; up to 7-11 dB at deep
nulls/very low levels. Diaphragm velocity cross terms similar. => current Float32 BLAB results are fine for SPL at
relevant levels; nulls, interface quantities and small cross-couplings are not reliable to 1 % at 20 kHz.
tol 1e-3 vs Float64 (kit/40): worst over 20 excitations BEM p 1.4e-3, q 1.9e-3, mech 4.2e-5, elec 5.3e-8; H(A) 0.26 GB,
H(C) 0.17 GB c64, build 30 + 8 s, 59-62 GMRES its, 12-15 s per excitation.

### 10 kHz trend (kit/41, Float64 reference: 291 s, +18 GB, +2.7 GB swap)
Engine Float64 A rows = conj(oracle) to 2.3e-15. tol 1e-3: worst BEM p 9.9e-4, q 1.6e-3, mech 3.2e-5, elec 2.8e-8;
tol 1e-4: 5.7e-5, 2.5e-4, 3.2e-6, 3.3e-9. GMRES 46-48 its (20 kHz: 59-62). H(A) 0.230/0.279, H(C) 0.206/0.244 of dense.
Archived A100 Float32 sweep (reference/20261120-conv_piece_h343) at 10 kHz vs Float64 field: within 6 dB of each
driver's max <= 0.016 dB, within 20 dB <= 0.082 dB, deepest points up to 1.5 dB.

## CUDA (RunPod A100 SXM 80GB, 2026-11-21, 18:25-19:33 UTC, 3.11 USD)
| test | build | other |
|---|---|---|
| A piece_h343 20 kHz f32 | 30 s (ACA 12.6 s, device 1.6 s) | GMRES 34 its 7 s, FP16 4.7e-4 |
| A piece_h245 f32 | 51 s | GMRES 76 its 22 s |
| A piece_h17 f32 | 101 s (ACA 66 s, device 8 s; recompress 19 s) | GMRES 86 its 53 s (host hmul; slower CPU than M1 Max) |
| A piece_h17 f64 | 107 s | identical results: Float64 kernels cost ~nothing on A100 |
| C piece_h343 / h17 f32 | 27 s / 85 s | driver RHS 2.6e-4 / 4.8e-4; C 2.25 GB vs 28.5 GB dense |
| real model (Float64 dump) cuda64 tol 1e-3 | A 29 s, C 11 s | worst BEM p 1.3e-3, q 1.9e-3, mech 4.1e-5, elec 5.4e-8; 59-62 its, ~21 s/exc |
| real model cuda64 tol 1e-4 | A 35 s, C 14 s | worst BEM p 1.6e-4, q 3.1e-4, mech 3.8e-6, elec 5.0e-9 |
Engine Float64 A rows vs CUDA oracle: 2.3e-15. Same numbers as the Metal runs. Bottleneck on CUDA: host side (ACA
bookkeeping, recompression, H*x on the CPU). Next: H*x and recompression on the device.
