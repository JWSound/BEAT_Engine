# Native phasor conventions

BEAT supports `exp(-i omega t)` and `exp(+i omega t)` explicitly. Unlabelled
requests retain negative time for compatibility; Boundary Lab requests positive
time. Select `solver_options.phasor_convention` for system solves and top-level
`phasor_convention` for source/Deploy solves and retained field evaluation.
Worker capabilities and response diagnostics make the convention observable.
A command and its request must select the same convention.

`BeatEnginePhasor.jl` defines the time derivative, Neumann scale, outgoing
propagation parameter and Burton-Miller coupling. Worker requests execute
serially inside `with_phasor_convention`, restoring the previous value on exit.
Low-level Julia callers must keep assembly, solve and evaluation inside the
same convention scope. Geometry-only caches are reusable. Retained Deploy
solution traces reject evaluation under another convention.

Positive time uses outgoing `exp(-i k r)`, `q=-i rho omega v`, BM `-i/k`, passive
FEM loss `+i eta k^2 M`, and electrical `s=+i omega`. Physical frequency and
wavenumber remain positive. CUDA/ROCm kernels receive signed propagation
parameters at host entry, without allocating conjugated dense operators.

CPU, CUDA, and Metal are qualified; Metal on an Apple M1 Pro through the Metal
arm of the native conjugation test. The ROCm arithmetic is wired but positive-time
worker requests are blocked until AMD hardware qualification. Legacy ROCm
requests remain supported.

The frozen negative-time fixtures are unchanged. `tests/reference_tests.jl`
is the required CPU gate. `tests/runtests.jl` adds accelerator qualification.
`tests/phasor_standalone.jl` runs focused native conjugation checks; setting
`BLAB_RUN_COUPLED_REFERENCE=1` adds complex electrodynamic excitations on both
sides of a weakly damped cavity pole, monolithic and condensed.

A convention change conjugates solutions for conjugated complex excitation
coefficients; it does not change conditioning, SPL, or dissipated power.
