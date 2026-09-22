# Deploy coupled feedback operator

The CUDA parity-ROM solver can assemble the frequency-dependent DP0-to-P1
Burton-Miller right-hand-side mapping once and apply it to each feedback vector
with a matrix-vector product. The existing regular, image, near and singular
quadrature paths build its columns, including the identity term and row weights.
Interleaved storage avoids a second full-sized complex matrix during conversion.
The operator is released at the end of each frequency; numerical operators are
never reused across frequencies or changed geometry.

`rom_feedback_mode` in an engine request, or `BEAT_DEPLOY_ROM_FEEDBACK` for the
worker environment, accepts:

- `auto` (default): a successful initial matrix-free solve measures the build and
  application costs. Subsequent matching workload profiles use the operator when
  those measured costs beat repeated integration with a 10% margin.
- `matrix_free`: use repeated integration, useful for reference comparisons.
- `cached_operator`: request the operator directly, still subject to memory checks.

Automatic calibration uses the calibration solve's actual operator-application
count, not a GPU model-specific threshold. Its workload profile includes node and
face counts and quadrature/correction sizes. Calibration is bounded to one profile
per worker and is only a performance estimate, never a numerical cache key. The
initial solve can take longer because it pays calibration and compilation.

The memory check reserves 25% over the matrix size plus 256 MiB for other work;
a concurrent allocation failure falls back to matrix-free evaluation. A
13,448-by-26,848 complex64 operator requires 2.69 GiB. Small-memory GPUs keep the
original path. Diagnostics report requested/effective mode and operator bytes.
Timers distinguish operator build, application and calibration costs.

## Validation

The phasor tests compare the assembled mapping with independent CPU operator
matrices and its product with direct CUDA RHS integration in both time
conventions. The standalone policy test covers measured wins/losses and
insufficient-memory cases; it is also part of `runtests.jl`. Run:

```text
julia --startup-file=no src/beat_engine/julia_local/tests/deploy_rhs_policy_tests.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_cuda src/beat_engine/julia_local/tests/phasor_standalone.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/reference_tests.jl
```

## Qualification: symmetry2026, 2026-09-22

Windows 11, Ryzen 7 5700X, RTX 2080 Ti (11 GiB), Julia 1.12.6 with 16 Julia/BLAS
threads and Python 3.13.13. Twelve cabinets (8 S218BP and 4 SKHORN), 13,448
boundary nodes and 26,848 faces, with rigid-ground and close-pair corrections.
The single-frequency field at 81.36 Hz contains 22,600 observation samples.

Three interleaved on/off repeats used the same BEAT checkout and updated Deploy
preparation, after priming both modes. Times include request preparation and
worker transport, excluding GUI rendering. The first three-frequency pass also
pays first-use sweep staging; ranges include it.

| Workload | Matrix-free median (range), s | Cached median (range), s |
| --- | ---: | ---: |
| Full single-frequency solve | 7.886 (7.818-7.986) | 4.032 (3.948-4.086) |
| Three-frequency sweep: 20, 81.36, 250 Hz | 23.538 (23.438-25.429) | 8.903 (8.878-8.979) |

Single-frequency feedback RHS time fell from about 4.4 s to 0.64 s, including
about 0.59 s to build the operator. Three-frequency RHS totals fell from about
16.4 s to 1.88 s, including about 1.68 s of operator construction. Worst relative
complex differences across repeated tests were 2.67e-6 for field pressure,
2.35e-6 for transducer velocity and 7.31e-7 for current. Worst magnitude error
within 30 dB of each reference peak was below 0.00019 dB. Both phasor conventions
also passed the small independent CPU-operator/CUDA comparisons.

A final real DeployWorkerClient run using automatic selection completed a full
100-frequency response sweep in 235.018 s and its warmed repeat in 233.162 s,
versus the original installed-engine baseline of 609.388 s. The combined Deploy
and BEAT changes reduced single-frequency latency from 13.372 s to 4.135 s and
plane-only updates to 0.357 s. These combined figures include Deploy preparation
and reuse changes; they are not engine-only speedup claims. This project has no
microphones: the sweep returns transducer/speaker responses, not 100 heatmaps.

In the updated full sweep, assembly took 91.322 s, coupled solving 137.075 s,
feedback RHS 59.437 s (including 54.482 s operator build), LU 62.089 s, and field
evaluation 0.108 s. RHS/build/LU are nested within coupled solving. Relative
transducer-velocity error against the original full sweep was 1.23e-6; voltage
matched exactly. Peak sampled total GPU memory was 5,728 MiB and Julia working
set 2,060 MiB; GPU totals include desktop usage. There was no progressive
operator retention across frequencies.

The raw complex results, timings and memory samples are retained as local study
artifacts; this project is not a portable fixture. These measurements qualify
this Windows/CUDA scene, not mesh convergence or performance on every GPU. No
mesh fidelity, quadrature or solver tolerance was reduced.
