# Exterior / coupled transducer qualification

## Criteria declared before execution

This qualification compares CPU Float64 exterior v3 requests with coupled v2
FEM–BEM–LEM requests containing only lossless air between the same moving solid
and a conforming outer interface. Both time phasors are exercised, with symmetry
off, density 1.21 kg/m³, sound speed 343 m/s, voltage 2.83 V, and frequencies
40, 160 and 400 Hz. Driver parameters are Re=6 Ω, Le=0.0005 H, Bl=7 N/A,
Mmd=0.02 kg, Cms=0.001 m/N, Rms=1.5 N s/m; motion is translation along +z.

Fine-level magnitude/phase budgets, fixed before any solve:

| Quantity | Magnitude difference | Phase difference |
| --- | ---: | ---: |
| Diaphragm velocity, current, V/i | 2% | 2° |
| Pressure at each nonnodal observation point | 3% | 2° |
| Radiation load diagonal | 3% | 2° |

Ratios use the exterior result as denominator. A separate complex difference
`max(abs(coupled/exterior - 1))` over frequencies, components/points and both
phasors must decrease with mesh refinement for each quantity. For the sphere,
the maximum over these quantities must also decrease with thinner air at each
mesh resolution. These are convergence requirements, not adjustable budgets.

The sphere radius is 0.1 m, with octahedral midpoint refinements 2 and 3
(128 and 512 triangles), and shell thicknesses 10 and 5 mm at both levels.
The box has half dimensions (0.1, 0.08, 0.06) m, two equal flat moving patches
on its +z face, and rigid remaining faces. Case (b) drives one patch and makes
the neighbouring half face rigid. Case (c) makes that neighbour a second
transducer, shorted through Ze. Each side has 2×2 then 4×4 squares
(48 then 192 triangles). Its homothetic whole-body shell extends 5 then 2.5 mm
above the top face. A whole shell provides a closed conforming FEM–BEM interface
without artificial walls at the patch perimeter. Only one patch is voltage
driven in each excitation; in case (c) its neighbour is shorted through Ze. Both
voltage basis columns are requested in case (c) to recover the full load matrix.

The sphere's geometric area deficit falls by approximately four on refinement;
the box geometry is exact and its P1 angular/tangential step halves. Shells use
one radial P1 layer split into conforming tetrahedra. At 400 Hz, kh is at most
0.074 for the sphere, and every angular edge is far below a wavelength. P1
geometric/interpolation error is expected to decrease with mesh step, while
finite-layer discretisation decreases as the layer thins. The budgets bound
agreement of the two discretisations, not absolute accuracy of either one;
the existing analytical sphere and frozen reference gates remain necessary.

For one driver, the coupled mechanical row implies
`Zload = Bl*i/u - Zm`, with
`Zm = Rms + s*i*ω*Mmd + 1/(s*i*ω*Cms)`, where s=+1 for positive time.
Equivalently `Fload = -fem_forceᵀ*p_fem` for this FEM-only moving surface.
For multiple drivers and all voltage basis columns, use
`Zload = (diag(Bl)*I - diag(Zm)*U) / U`, where rows of U/I are drivers and
columns are voltage excitations. This recovers each diagonal without assigning
mutual force to a self load. No sealed chamber, orbit, completion or area
transformer is present.

Execution results and limitations will be recorded after the fixed gates run.

## Observed outcome: partial agreement, box qualification failed

The standalone gate took 43.7 s (48 s including startup), so it is registered
last in `tests/runtests.jl`. It passed 348 assertions and failed eight fixed
magnitude-budget assertions. All refinement checks and both sphere thickness
checks passed. Both phasors give the same error magnitudes.

Fine-level maxima across all frequencies/points/drivers and both phasors:
entries are magnitude difference (%) / phase difference (degrees).

| Case / shell | u | i | V/i | Pressure | Load diagonal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sphere, 10 mm | 0.173 / 0.049 | 0.264 / 0.095 | 0.265 / 0.095 | 0.794 / 0.049 | 1.343 / 0.013 |
| Sphere, 5 mm | 0.116 / 0.032 | 0.176 / 0.063 | 0.176 / 0.063 | 0.640 / 0.032 | 0.892 / 0.005 |
| Box, single patch | 0.374 / 0.105 | 0.567 / 0.298 | 0.571 / 0.298 | 2.293 / 1.111 | **8.637** / 0.756 |
| Box, shorted neighbour | **3.161** / 1.393 | **3.161** / 1.393 | 0.569 / 0.298 | 2.330 / 1.127 | **8.637** / 0.778 |

The sphere meets all declared criteria. The box load magnitude decreases from
about 20% to 8.64% disagreement but exceeds 3% at the fine level. The shorted
neighbour's u/i magnitude difference decreases from 5.40% to 3.16% but exceeds
2%. Its worst magnitude discrepancy is at 160 Hz; the load discrepancy is at
400 Hz. This is a failed qualification, not evidence assigning a solver defect:
these two levels do not separate remaining edge/interpolation error from a
solver discrepancy. Investigation stopped without changing solvers, baselines
or criteria. The failing assertions remain visible in the default runner.

Limits: small closed polyhedral fixtures, CPU Float64, symmetry off, selected
frequencies/points, real voltage basis and shorted termination only. The load
checks bound complex magnitude/phase, not radiation resistance separately near
low ka. Arbitrary geometry, finer box convergence, shell thickness convergence
independent of mesh refinement for the box, other terminations, symmetry and
accelerators are unqualified. No performance improvement is claimed.

## Revision decided after the first run — 2026-10-07

The original 48/192-triangle qualification failed as recorded above. The revision
was explicitly decided **after that first run and the subsequent diagnostic**;
it is not a retroactive pass of the original preregistered fixture levels.
The diagnostic reproduced the gap with ideal sources, while transducer load and
field operators matched ideal-source operators to about 1e-12. In the fixed-mesh
thin-shell limit, the coupled solution matched an independent continuous P1
projection of the sharp moving/rigid face flux. That continuous nodal interface
flux cannot represent the patch step on a coarse mesh. Refinement reduced the
gap; both box cases at 3,072 triangles met the original fine budgets. Shell
thickness also contributes, with homothetic offsets a smaller contributor.

The revised criteria, declared before executing the revised gate, retain the
same box, homothetic shell, driver, 40/160/400 Hz, four observation points,
CPU Float64, symmetry off, quadrature 4 and both phasors. Only the box levels
change: 8×8 and 16×16 squares per side, **768 and 3,072 triangles**, with top
shell thicknesses **1.25 and 0.625 mm**. The sphere cases are unchanged. The
single-driver neighbour remains rigid; the paired case retains both voltage
basis columns, with the other driver shorted in each column.

Fine budgets remain u/i/V/i ≤2% magnitude and ≤2° phase, and pressure/self load
≤3% magnitude and ≤2° phase. Every quantity's maximum complex error must
strictly decrease from 768 to 3,072 triangles, separately for each phasor/case.
No solver, tolerance, frozen numerical baseline or extraction map is changed.

An ideal-source control now uses the same physical box patches in both paths,
with both unit velocity basis columns for the pair. Coupled ideal sources use
the supported normal-velocity profile, whose inner normal is -z; negating the
returned pressure gives physical +z unit velocity on this flat patch. This does
not qualify coupled translation sources on arbitrary surfaces. Pressure is
integrated over the inner physical patch as `F = Σ A mean(p)`. Actual transducer
load and field operators are `F/U` and `Pᵀ/U`; load is also checked against
`(Bl I - Zm U)/U` and the exterior public load matrix. Each actual load/field
operator must agree with the ideal operator in relative Frobenius norm ≤1e-9.
Exterior ideal integration must agree with both public load quantities. Ideal
self-load and field path gaps also obey the unchanged fine budgets and decrease
with refinement. These controls distinguish a future network failure from a
path/discretisation discrepancy without another diagnostic study.

Signed resistance and reactance are reported per frequency, case, path and
phasor, with **no separate budget and no resistance qualification**. The
complex-load magnitude/phase budget does not establish relative accuracy of a
small resistive part.

Runtime target: ≤10 minutes for the standalone qualification, broker expected
≤7 minutes with a ≤14-minute wall limit. Registration will be decided from the
actual revised gate: retain in `runtests.jl` only if it fits approximately two
minutes; otherwise run it as a separate mandatory numerical qualification gate,
like `reference_tests.jl`. Revised execution results will be appended below.

The revised gate exceeded two minutes during measurement, so its former default
runner registration has been removed. It is a **required standalone numerical
qualification gate**, to run alongside the full `runtests.jl` and
`reference_tests.jl` gates. Submit this command through the compute broker:

```sh
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/exterior_coupled_agreement_tests.jl
```

Set `BEAT_AGREEMENT_REPORT` to a writable JSON path to retain aggregate metrics
and per-frequency signed load diagnostics. Stdout additionally preserves complex
actual and ideal-source samples. The gate temporarily uses one BLAS thread,
matching the diagnostic, and restores the ambient setting afterwards.

### Revised execution results — 2026-10-07

Broker gate `261007-215036-compute-1588` exited 0: **23,724 passed, zero
failures**. It made 44 public requests, yielding 132 frequency results and 180
excitation RHS columns. Measured qualification work was 512.268 s; the testset
took 8m33.1s, and broker elapsed time including startup was **8.66 minutes**.
Expected standalone runtime is approximately **8–10 minutes** on the measured
CPU configuration (two Julia threads, one BLAS thread), meeting the ≤10-minute
target. This is a gate-sizing measurement, not a performance comparison.
The gate is unregistered from `runtests.jl` and remains separately required.

All unchanged sphere criteria passed, including both thickness checks. Every
actual and ideal box quantity's complex error decreased at refinement, separately
for both phasors. The table below gives maxima across all frequencies,
drivers/points and both phasors, as **magnitude % / phase ° / complex %**:

| Box case | Triangles | Top h (mm) | u | i | V/i | Pressure | Self load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| box | 768 | 1.25 | 0.135657 / 0.038240 / 0.136775 | 0.206598 / 0.106924 / 0.278274 | 0.207025 / 0.106924 / 0.278850 | 0.827982 / 0.407939 / 1.023704 | 3.021722 / 0.242613 / 3.050359 |
| box | 3072 | 0.625 | 0.048837 / 0.013786 / 0.049301 | 0.074425 / 0.038460 / 0.100207 | 0.074480 / 0.038460 / 0.100282 | 0.297758 / 0.149040 / 0.368448 | 1.075809 / 0.088715 / 1.086776 |
| box_pair | 768 | 1.25 | 0.788807 / 0.337286 / 0.844903 | 0.788807 / 0.337286 / 0.844903 | 0.207227 / 0.107116 / 0.279122 | 0.834188 / 0.412109 / 1.032999 | 3.031884 / 0.244018 / 3.060752 |
| box_pair | 3072 | 0.625 | 0.186092 / 0.083311 / 0.202084 | 0.186092 / 0.083311 / 0.202084 | 0.074953 / 0.038545 / 0.100733 | 0.298054 / 0.150269 / 0.369958 | 1.081177 / 0.088715 / 1.092077 |

The ideal-source fine self-load maxima were 1.075809% / 0.088715° for the
single patch and 1.081177% / 0.088715° for the pair. Fine ideal-source field
maxima were 0.346427% / 0.152906° in both cases. The maximum relative
transducer/ideal operator difference across all cases, levels, paths, frequencies
and phasors was **4.259e-13 for load** and **5.095e-13 for field**, below 1e-9.
Pressure-integrated and electromechanically implied loads differed by at most
2.393e-14. Both public exterior ideal-load comparisons passed.

Resistance and reactance are retained separately in the gate's
`AGREEMENT_CONTROL_SAMPLE` output and optional JSON report, for each frequency,
path, level, case and phasor. They have no budgets; **radiation resistance is
not qualified**. At 40 Hz, fine single-patch R was 0.004528032 exterior versus
0.004529661 coupled N s/m, while X was 0.228808208 versus 0.226451967 N s/m
for positive time. These are diagnostics, not a separate accuracy guarantee.

The original failed result remains the outcome of the original 48/192 levels.
This revised pass qualifies only the specified refined fixtures and criteria.
It does not establish absolute accuracy, arbitrary geometries, independent
box shell-thickness convergence, other terminations, symmetry, Float32 or
accelerators. No solver change or performance improvement is claimed.
