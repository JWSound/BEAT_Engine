# Deploy feedback compression research spike

This opt-in numerical experiment compresses the DP0-flux to P1-pressure
Burton–Miller feedback operator `R` in the Deploy CUDA parity-ROM path.
The dense exterior `L` matrix, its CUDA LU, the package ROM, and the ordinary
solver defaults are unchanged. This is a block-low-rank feasibility prototype,
not a hierarchical LU implementation or a production speed claim.

## Request

Use a validated `boundary_lab_deploy_rom` request with:

```json
{
  "rom_feedback_mode": "compressed",
  "experimental_rhs_compression": {
    "tolerance": 0.0001,
    "leaf_size": 128,
    "admissibility": 1.0,
    "survey_only": false,
    "report_path": "absolute-path-to-rank-report.json"
  },
  "include_complex_pressure": true
}
```

The existing `BEAT_DEPLOY_ROM_FEEDBACK` override also accepts `compressed`.
This mode requires CUDA and a ROM request. Set `survey_only` to assemble four
spatial column slabs and inspect up to eight row leaves per slab; the actual
solve then uses exact matrix-free feedback. Sampled storage is not an estimate
of total operator storage.

## Numerical implementation

* `assemble_burton_miller_rhs_cuda(...; assemble_operator=true,
  operator_columns=indices)` assembles exact selected columns. All existing
  singular, near, reflected-image, identity and row-weight contributions use
  the same scatter mapping. No full `R` allocation is necessary.
* Source-face centers and target nodes are split geometrically. Admissibility
  uses full triangle support, including every triangle supporting a P1 node,
  and checks both direct and reflected distances: maximum support diameter must
  be no larger than `admissibility` times either separation distance. Increasing
  this setting admits more blocks for compression but does not relax the error
  tolerance; overlapping supports remain dense.
* Near blocks remain dense. Admissible row clusters are merged over column
  leaves; complete-pivot cross approximation on each exact dense block stops
  only after its actual Frobenius error is small enough. A failed rank or
  storage-benefit check retains the exact dense block. The rank cap is 64.
* Dense and low-rank blocks are packed into GPU arrays. Two kernels apply the
  right factors and then the dense/left factors, avoiding per-block launches.
  Complex outer products do not conjugate their right factors.
* After GMRES, an independent exact-quadrature feedback application and the
  original LU audit the **exact left-preconditioned** equation. A relative
  residual above `1e-3` rejects the experiment. This is not the unpreconditioned
  physical-equation residual and is not a mesh-convergence test.

The initial drive RHS remains exact. Final pressure feeds the unchanged ROM
velocity/current reconstruction and exterior field evaluator. New settings
must be included in any future application-level solution cache key; the
research harness always submits a fresh solve and bypasses field-only reuse.

## Validation and measurement

Run the standalone CUDA gate:

```powershell
$env:BLAB_BEAT_ENGINE_GPU_BACKEND='cuda'
julia --project=src/beat_engine/julia_cuda --threads=4 src/beat_engine/julia_local/tests/deploy_rhs_compression_tests.jl
```

Also run the required CPU `tests/reference_tests.jl` entrypoint under
`julia_local`, following the repository contributor guide. Do not update frozen
baselines to accept this approximation.

The Deploy checkout contains `scripts/research_deploy_compression.py` and
`scripts/compare_deploy_compression.py`. The replay harness validates a saved
schema-12 scene and package hashes, preserves channel DSP, prepares a real ROM
request, and records source revisions/diffs/snapshots, raw events, complex
results, timings, and sampled device-wide GPU memory. It uses the imported
engine paths and disables bundles so edited source is actually executed.
It does not use the Boundary Lab `.blab.json` CLI for Deploy scenes.

Example from the Deploy checkout (adjust paths for another machine):

```powershell
$env:PYTHONPATH='E:/Code/boundary_lab_deploy/src;E:/Code/BEAT_Engine/src'
python scripts/research_deploy_compression.py 'G:/My Drive/Projects/Boundary Lab Projects/largebattlehawkscene.blabdeploy.json' --output runs/compression-study --counts 16 --modes matrix_free,compressed --leaf-size 128 --admissibility 4 --warmup 1 --repeat 3 --field-planes
python scripts/compare_deploy_compression.py runs/compression-study
```

Use a new output directory for each experiment. `--prepare-only` validates and
stages requests without starting a worker. `--backend cpu --counts 1` supplies
a tractable real-project CPU reference; compression itself is CUDA-only.
`--field-planes` evaluates both saved planes against each mode's final retained
boundary solution without repeating the coupled solve. `--frequencies` accepts
exact exported package frequencies, not arbitrary interpolated ROM frequencies.

Compare at least three interleaved measured runs after warming both modes.
Include construction, packing, audit and output costs in end-to-end time;
do not infer a speedup from apply-only timings. A pair passes the initial
comparison gate only with matching finite outputs, both GMRES residuals at or
below `1e-4`, the independent audit at or below `1e-3`, complex relative L2
errors below 1%, and worst magnitude errors below 0.1 dB within 30 dB of peak.
The harness's `wall_s` measures the worker request, including its repeated
geometry setup and transport. Shared Python request preparation is reported
separately as `prepare_s`; worker startup is outside that timer. Do not label
`wall_s` as desktop click-to-display latency.

## Deliberate limitations

Construction still computes every dense entry and performs host compression;
it is quadratic work, not an ACA entry-oracle implementation. Near corrections
are recomputed for each slab. Packed application is a simple bandwidth-oriented
kernel, not batched BLAS or an H2 nested-basis algorithm. Factors are rebuilt
per solve, and global LU remains. VRAM checks reject oversized operators instead
of silently enabling oversubscription. Host peak memory is not measured by the
current harness; GPU samples include other applications and can miss brief peaks.

This approximation changes numerical results and should be reviewed in a
separate experimental numerical PR before any default or packaging changes.
