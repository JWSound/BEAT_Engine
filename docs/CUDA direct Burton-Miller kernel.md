# CUDA direct Burton-Miller kernel

The direct Burton-Miller assembly (`assemble_burton_miller_neumann_system_cuda`
and `assemble_burton_miller_rhs_cuda`) integrates every regular element pair
with `_cuda_bm_fused_regular_kernel!` (`src/BeatEngineCudaRegular.jl`). The
earlier path ran the four-operator `_cuda_regular_kernel!` in direct mode: it
accumulated the single layer, double layer, adjoint double layer and
hypersingular operators separately (48 floats per element pair) and combined
them only in the scatter.

## What the fused kernel does

The Burton-Miller combination is linear, so it is applied per quadrature point
pair, as Metal's fused assembler does:

$$
\mathrm{lhs}_{ab} = \sum t_a r_b \left[-\partial_{n_y} G - i k\,(n\cdot n')\,G\right] w
+ \frac{i}{k}\,\mathrm{curl}_{ab} \sum G\, w ,
\qquad
\mathrm{rhs}_a = \sum t_a \left[-G - \frac{i}{k}\,\partial_{n_x} G\right] w .
$$

- Per test point, the trial quadrature points fold into 10 live floats: three
  left-hand-side partial columns, one right-hand-side partial and `sum G w`. They
  are expanded by the test basis once per test point into the 18 + 6 pair
  accumulators.
- The curl term is loop-invariant. It is added once from `sum G w`, after the
  loop, so its nine products are not live during it.
- The trial quadrature points are formed once per element pair instead of once
  per point pair. The rule size is a compile-time constant (`Val(R)`), so the
  trial fold unrolls and its points stay in registers.
- `1/(4 pi)` is folded into the per-pair Jacobian, which removes a
  full-precision division from every point pair.

Results match the four-operator kernel to float32 summation order.
`BEAT_CUDA_BM_REGULAR_KERNEL=generic` selects the previous kernel for
comparisons. The operator-matrix path and the coupled FEM-BEM path are
unchanged.

## Validation

`runtests.jl` compares the fused and generic kernels on `sample_quarter.msh`
with `xy` images and image-singular corrections, for the system and the
right-hand-side-only launch. It does this for every rule size production uses
(order 1, 2 and 4: 1, 3 and 6 points) at relative tolerance 2e-5. The phasor
tests repeat the comparison in both time conventions through an `x`-image launch
(every tetrahedron face pair is adjacent, so only image launches run regular
pairs), including the 2-D right-hand-side launch Deploy's feedback operator uses.

## Qualification: loudspeaker model, 2026-10-01

The request is the same as in [Exterior CUDA excitations](Exterior%20CUDA%20excitations.md):
44,662 faces, 22,549 P1 dofs, `x` symmetry, 60 excitations, 6-point rule, Float32,
20 kHz, run with `scripts/benchmark_worker.py`. Both kernels used the cached
right-hand-side mapping. The runs were interleaved, with a fresh worker for each.

Tesla T4 16 GB (`standard-v3-t4`, driver 535.247, CUDA.jl 6.2), this branch against
its parent:

| Kernel | Runs | Assembly, s | Solve, s | Wall, s |
| --- | ---: | ---: | ---: | ---: |
| four-operator (parent) | 3 | 17.86 (17.62-17.87) | 8.12 (8.05-8.13) | 27.6 (27.3-27.6) |
| fused | 3 | 12.56 (12.42-12.57) | 8.18 (8.11-8.18) | 22.3 (22.1-22.3) |

Assembly is 30% faster. Every run agreed with `main`, run on a V100, to
relative L2 5.7e-7 over the 60 x 290 complex pressures (worst excitation
1.2e-5, worst level error 0.00005 dB), and to 1.4-1.7e-7 on the impedances. The
parent shows the same agreement, and it is the float32 atomic-order noise seen
between any two runs.

On a Tesla V100-SXM2 32 GB, with the `sincos` change also applied, assembly
went from 5.68 s (4 runs, 5.67-5.68) to 4.55 s (2 runs, 4.53-4.58), 20% faster.
With one excitation (no mapping) it went from 5.50-5.53 s to 4.12 s. Against a
Float64 CPU reference, the fused kernel's pressures are within relative L2
5.62e-5 at 20 kHz, the same as `main`'s.

The kernel is no longer limited by quadrature arithmetic: a 3-point rule (a
quarter of the point pairs) assembled in 4.15-4.30 s against 4.45 s on the V100.
The remaining time is most likely the atomic scatter of about 24 float atomics
per element pair. That has not been profiled yet.
