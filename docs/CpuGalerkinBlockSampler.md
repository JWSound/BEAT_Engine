# Experimental CPU Galerkin block sampler

This is the first operator-construction component for a compressed BEM path.
It is opt-in: existing exterior and coupled production solves do not call it.
It is not an H-matrix compressor, an iterative coupled solver, or a GPU backend.

The implementation is based on `memory-fork` commit
`4992604bee3f7fcab29c99f7d2467ef62284cc18`.

## Recovery provenance

The original download archive was unavailable after its temporary workspace
disappeared. The sampler source was restored from the complete source text
retained in the conversation. The focused sampler tests and this guide were
reconstructed, so this is a newly packaged and newly tested recovery version,
not a claim of byte-for-byte identity with the unavailable sampler patch.
The archive's `validation.json` and fresh logs contain this version's results.
Earlier test logs have not been recreated or presented as recovered evidence.

## Purpose

The memory fork already reduces dense storage with compact flux columns,
sparse identities, split precision, and spillable residual storage. Its large
pressure matrix is still dense. Hierarchical compression needs access to small
Galerkin blocks without first allocating that full matrix.

`BeatEngineCpuBlockSampler.jl` reuses the existing scalar CPU pair kernels to
sample the combined Burton-Miller operators:

```text
A = 0.5 Mpp - D + alpha H
C = S + alpha (adjD + 0.5 Mpq)
```

It preserves the P1 pressure basis, DP0 flux numbering, singular quadrature,
reflected geometry, surface-curl transformation, and symmetry row weights.
The current phasor convention is captured at construction. The regular pair
kernel's RHS block represents `-C`, which the adapter converts to `C` before
adding its mass term.

## API

These functions are available as qualified `BeatEngineCore` functions; they
are not added to the production solver configuration or worker protocol.

```julia
sampler = BeatEngineCore.build_cpu_galerkin_block_sampler(
    mesh, p1, dp0, k, rule;
    singular_order=3,
    symmetry_mode=:xy,
    identity_rule=rule,
    near_correction_cache=nothing,
    image_near_correction_cache=nothing,
)

Ablock = BeatEngineCore.sample_burton_miller_block(
    sampler, pressure_rows, pressure_columns; operator=:A,
)

Cblock = BeatEngineCore.sample_burton_miller_block(
    sampler, pressure_rows, flux_columns; operator=:C,
)

BeatEngineCore.sample_burton_miller_block!(
    destination, sampler, pressure_rows, pressure_columns; operator=:A,
)
```

Use the same prepared mesh, spaces, positive physical wavenumber, quadrature
and symmetry as the reference solve. `Float32` and `Float64` are supported.
`k` and the rules must have the mesh precision. The separate `identity_rule`
supports coupled paths whose mass quadrature differs from operator quadrature.

Selections are one-based integer vectors or ranges. Repeated, reordered and
empty selections follow normal Julia matrix-indexing semantics. Output element
type must match `Complex{T}` for a mesh with scalar type `T`.

## Supported scope and ownership

- Vertex-based continuous P1 and face-based DP0, including DP0 permutations.
- Symmetry `:off`, `:x`, and `:xy`. Ground-plane mode is explicitly rejected.
- Duffy treatment for direct and reflected coincident/edge/vertex pairs.
- Existing host near-correction caches, including tuple/vector collections and
  different correction orders per pair. Each transform must be active in the
  selected symmetry mode.
- Complex operators with either supported phasor convention.

Input geometry, spaces and caches must belong to the same prepared model and
remain unmodified. Shape, pair bounds, rule indices and transform checks do not
constitute a geometry provenance check. The sampler does not support GPU/device
near caches and does not silently ignore unsupported corrections.

## Memory and arithmetic limits

No global dense BEM or mass matrix is allocated. Preprocessing stores element
and quadrature data, incidence lists, local singular-pair data, and sparse
mass terms. Existing adjacency helpers use incidence and geometric hashing;
there is no general all-pairs scan during this setup. Pathological meshes can
still have large adjacency candidate sets.

Near-correction quadrature currently retains data for all faces for each rule
and image. This has a potentially significant linear memory constant; a future
production implementation can restrict it to faces actually participating in
the corrected pairs.

The requested output block can itself be large. The caller must impose its own
block-size budget. This adapter does not provide a whole-matrix `mul!`, and
requesting every row and column would deliberately allocate a dense output.

The sampler directly evaluates a Duffy value for an image-singular pair, while
the existing dense image path adds a regular value and then a Duffy-minus-regular
correction. These are algebraically identical but can round differently.
Agreement is numerical, not a claim of bitwise equality.

## Validation

Run from the repository root after instantiating the CPU Julia environment:

```sh
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local \
  src/beat_engine/julia_local/tests/cpu_block_sampler_tests.jl

julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local \
  src/beat_engine/julia_local/tests/reference_tests.jl
```

The focused suite uses the unchanged four-operator and fused CPU assemblers as
independent discrete-algebra oracles. It covers both precisions and phasors,
off/X/XY symmetry, DP0 permutations, block selections, real and complex
multi-RHS products and small exterior solves, near and image corrections,
and separate identity quadrature. It is included in both test runners.

Passing these tests is not a qualification of maximum pointwise acoustic
error on a user model. No production speedup, peak-memory reduction,
20 kHz user-mesh accuracy, or Metal/CUDA qualification is claimed.

## Next components

1. Build cluster/support-box and compression adapters around this sampler.
2. Implement compressed operator application for A and compact C.
3. Apply FEM/interface/LEM coupling without rebuilding a dense final system.
4. Add an independently controlled residual operator and robust iterative
   driver, with a memory-bounded fallback policy.
5. Measure ranks, build cost, total memory and complex output errors on actual
   models at 1, 5, 10 and 20 kHz before choosing plain ACA versus directional
   high-frequency compression and before adding FP16 storage or GPU kernels.
