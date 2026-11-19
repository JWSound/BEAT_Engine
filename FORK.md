# BEAT Engine – memory fork

This is an unofficial fork of [JWSound/BEAT_Engine](https://github.com/JWSound/BEAT_Engine).
It reduces the memory and time of coupled FEM–BEM solves (the solver behind
Boundary Lab), so that large models up to 20 kHz fit on machines such as a
32 GB Apple Silicon laptop. Everything lives on the `memory-fork` branch, based on
upstream `f25cd8e`. `main` is an unchanged mirror of upstream.

All changes are opt-in or result-identical. Default behaviour matches upstream,
except for lower memory use.

## What changed

| Area | Change | Switch |
|---|---|---|
| Coupled dense memory (Metal + CUDA) | Compact C columns, no host copies of A/C, sparse identities | always on |
| Dense solve | Float32 split storage + Float64 iterative refinement, GMRES-IR fallback, disk spill | always on |
| Metal index limit | 64-bit dense indices; lifts the ~32k-node cap ("Int32 dense indices") | always on |
| Metal buffer limit | Column-panel assembly when A exceeds one Metal buffer or a quarter of the GPU working set; one panel in flight at a time | `BLAB_METAL_COUPLED_PANEL_COLUMNS=auto\|<n>\|off` |
| LU library | Choose OpenBLAS or Apple Accelerate for the dense LU | `BLAB_DENSE_LU_LIBRARY=blas\|accelerate\|auto` |
| Dense build (Metal) | The dense system takes over A's host memory (restrided in place) instead of copying it: A and the dense system never coexist. Saves 8 n² bytes at the build peak when the dense spill is active (large models) | `BLAB_METAL_COUPLED_ADOPT_A=on\|off` |
| Threads in containers | "auto" Julia threads and OpenBLAS threads follow the cgroup CPU quota (rented GPU pods) | `OPENBLAS_NUM_THREADS` overrides |
| Interface radiation | Skip the interface-radiation replay (a second LU per frequency) when the decomposition is not needed | `BLAB_COUPLED_INTERFACE_RADIATION=off` |
| Experimental | Dense GMRES (Float32 inner, Float64 outer) with automatic LU fallback | `BLAB_COUPLED_DENSE_SOLVER=gmres` |

## Measured on M1 Max, 32 GB (10 kHz quarter-horn model, N_B ≈ 23k)

| Run | Wall time | Memory |
|---|---|---|
| Upstream | 276 s | ~27 GB (incl. 5.6 GB swap) |
| Fork | ~200 s | 21.6 GB |
| Fork, interface radiation off | ~118 s | 16.2 GB |

Adopting A instead of copying it (patch 13), same model, dense spill forced to emulate the
large-model regime: footprint 17.3 → 11.9 GB, results bit-identical; with defaults (no spill)
15.2 GB.

Single runs on one machine, not yet medians of interleaved runs as `docs/Benchmarking.md`
asks; treat them as indicative. All other result quantities stay bit-identical. A 54.6k-node model at 20 kHz now
assembles on Metal; it failed upstream.

The dense GMRES option does not win yet on these models: it needs too many iterations
without a better preconditioner. It stays opt-in and falls back to LU when it does
not converge. See `docs/Memory Fork Summary.md` for details and future directions
(frequency-recycled preconditioning, FMM / H-matrix).

## Status

- Metal: validated on M1 Max (tests and A/B runs against upstream).
- CUDA: validated on an A100 SXM 80 GB pod: the HF piece matches the Metal result within 0.05 dB
  in the main lobe (the CUDA path keeps upstream's `interface_elimination = none`); full quarter
  model at 20 kHz (n = 59-63k): ~3.5 min per frequency, 50-57 GB GPU memory.
- Metal: a 54.6k-node quarter model at 20 kHz now assembles and factors on a 32 GB M1 Max (it
  failed upstream), but swaps: more than 2.5 h per frequency.
- Testing on M4/M5 (Accelerate with SME) is especially welcome.

## License

GPL-3.0, same as upstream. The changes are marked in the commit history.
