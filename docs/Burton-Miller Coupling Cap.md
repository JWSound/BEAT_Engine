# Burton-Miller coupling cap

The exterior Neumann solve couples the conventional and hypersingular
equations with `eta`. It was the bare `eta = s i/k` (`s` the propagation sign,
see `Phasor Convention.md`). As `k -> 0`, `|eta|` grows without bound, the
`eta H` term swamps `0.5 I - D`, and the bottom of a sweep is the worst
conditioned part of it. The coupling is now

    eta = s i |k| / max(k^2, c),   c = 1/R^2

where `R` is half the diagonal of the body's bounding box, completed across the
active symmetry planes (and across the ground plane in Deploy). Above `kR = 1`
the expression is `s i/k` to the last bit. Below that point `|eta| = k R^2 < R`.
`BLAB_BEAT_BM_COUPLING_CAP` takes `auto` (default), `off`, or an explicit `c`
in 1/m^2.

The reasoning, from m3gnus's original change (0727e55): under Galerkin testing
`H` is an order +1 operator with scale `1/R` on a body of size `R`, and
`0.5 I - D` is order 0. Balancing the two wants `|eta| ~ R`, and the bare `1/k`
exceeds `R` exactly when the body is acoustically compact. Capped couplings of
this form are studied in https://arxiv.org/abs/2405.10722. On a sphere of
radius `a` this `R` is `sqrt(3) a`, so the cap engages below `ka = 1/sqrt(3)`.

## What it changes

Any `eta` with the correct sign gives a uniquely solvable equation with the
same exact solution, so the cap changes discretisation error, not the physics.
It does change solver output below `kR = 1`, on every backend, so it is a
separate change from the hardware-backend work.

The test is an off-centre point source inside the unit sphere, judged against
the exact field, in Float64:

| Mesh | k | Field error, uncapped | Field error, capped | cond, capped/uncapped |
|---|---|---|---|---|
| 128 faces (66 P1 dofs) | 0.02 | 3.2307e-2 | 3.2218e-2 | 0.094 |
| | 0.20 | 3.1961e-2 | 3.1919e-2 | 0.241 |
| | 0.45 | 3.0844e-2 | 3.0837e-2 | 0.668 |
| | 0.70 | 2.9679e-2 | 2.9679e-2 | 1.000 |
| 512 faces (258 P1 dofs) | 0.02 | 7.9309e-3 | 7.9205e-3 | 0.041 |
| | 0.20 | 7.8421e-3 | 7.8369e-3 | 0.171 |
| | 0.45 | 7.5579e-3 | 7.5571e-3 | 0.626 |
| | 0.70 | 7.2698e-3 | 7.2698e-3 | 1.000 |

The radiated-field error improves by at most 0.3% of itself, which is noise
next to the discretisation error, and `k = 0.70` (above engagement) is
identical. The surface pressure trace moves by up to about 0.2% of its own
error, in either direction (6.9324e-2 -> 6.9412e-2 at `k = 0.02` on the coarse
sphere). The large effect is conditioning, 10-25x at the lowest `k`.
m3gnus measured the far field on the ATH ladder moving by at most 0.008 dB
main-lobe rms where the cap is active.

Where it pays depends on the solver. Main solves exterior systems with LU, where
better conditioning matters only through Float32 round-off. The iterative paths
gain most: Deploy's CUDA GMRES, and the adaptive dense solve in the Metal
backend PR. On m3gnus's ATH ladder at tolerance 1e-5, GMRES iterations dropped
80 -> 44 (A1, 100 Hz), 132 -> 78 (A1r, 100 Hz) and 78 -> 56 (A5, 100 Hz). On the
bundled sample at 49 Hz, the Metal branch's GMRES gate converges in 33 capped
and fails to converge in 47 uncapped.

## Coverage

Every Burton-Miller site derives `c` from the body and passes the same value to
both sides of the system:

- CPU and ROCm operator paths, and CUDA's operator path (`solver.jl`,
  `solve_burton_miller_neumann`).
- CUDA direct assembly and its RHS-only variant, which is the default exterior
  CUDA path (`BeatEngineCudaBurtonMiller.jl`).
- The coupled FEM-BEM solver, with and without the combined CUDA kernels
  (`BeatEngineCoupled.jl`, `BeatEngineCudaCoupledBurtonMiller.jl`), and its
  exterior solve (`coupled_solver.jl`).
- Deploy: direct, ROM feedback, CUDA operator and CPU paths.

The kernels already combined `-D + i s H` and `-S - i s K'` with the signed
outgoing wavenumber. They now receive
`copysign(burton_miller_coupling_scale(k, c), k)` in place of `inv(k)`, which
keeps the sign and is `inv(k)` exactly when `c = 0`. Every helper defaults to
`inv(k)`, so callers that pass no cap are unchanged.

Metal is not on main. Its fused kernels need the same `copysign` treatment when
the Metal backend lands. Taking the magnitude unsigned there silently flips the
coupling under positive time, which the Metal arm of `tests/phasor_standalone.jl`
catches (25/28 instead of 28/28).

## Gates

- `tests/runtests.jl`, "burton-miller coupling parameter": sign under both
  conventions, uncapped bit-identical to `s i/k`, capped continuity and bound,
  body radius across symmetry, override parsing.
- `tests/reference_tests.jl`, "capped Burton-Miller coupling against the exact
  exterior field": the sphere study above under both conventions. The capped
  field error is within 0.1% of uncapped or better below engagement, identical
  above it, and the capped system is better conditioned. The frozen reference
  fixtures pass unchanged with the cap on.
- CUDA: none of the CUDA changes have run on hardware. They need
  `hardware.yml` with `backend=cuda` (or `tests/runtests.jl` on a CUDA host)
  before merge.

## Open questions

1. Default. `auto` changes every user's low-band output by default. `off`
   keeps results bit-identical and makes the cap opt-in.
2. Choice of `R`. Half the bounding-box diagonal is geometry-derived but
   anisotropic: a long thin body gets a large `R` and a low engagement
   frequency. The balance argument only fixes `R` up to an O(1) constant.
3. Deploy includes the ground image in `R`, matching how symmetry planes are
   treated. A half-space body could argue for its own extent instead.
4. Timing relative to the Metal backend. The cap's main payoff is on the
   iterative solves, and the adaptive dense solve arrives with that PR.
