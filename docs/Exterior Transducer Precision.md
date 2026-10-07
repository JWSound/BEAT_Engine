# Exterior transducer precision qualification

CPU direct-system assembly, quadrature and singular order 4, on the exact slice-2 oscillating sphere (radius 0.1 m, 512 faces, 258 nodes). Float32 is compared with Float64 using the same faceted geometry. This is a measurement report, not a numerical baseline.

LEM: Re=6 ohm, Le=0.5 mH, Bl=7 N/A, Mmd=15 g, Cms=0.5 mm/N, voltage=2.83 V. Rms=1 or 0.6 N s/m gives bare mechanical Q=sqrt(Mmd/Cms)/Rms=5.48 or 9.13; electrical and radiation loading reduce the driven resonance Q.

Frequencies (Hz): 20, 35, 40, 45, 48, 50, 52, 54, 56, 58, 60, 62, 65, 70, 80, 100, 600. The 2 Hz spacing around resonance resolves the bare mechanical resonance at 58.1 Hz and the radiation-loaded resonance below it. Pressure points (m): (0,0,0.15), (0,0,2), (1,0,2).

Each entry reports the maximum over frequencies and, for pressure, all three points. Complex relative error is pointwise abs(test/reference - 1); dB and degrees are absolute amplitude-ratio and principal phase-ratio differences. Both phasor conventions were measured. LU refinement requests up to three Float64-residual corrections of the rounded Float32 operator. Every refined solve stopped after one correction. It cannot recover assembly or geometry error.

| Phasor | Q | Refinement requested | Quantity | Max complex relative | Max dB | Max degrees |
|---|---:|---:|---|---:|---:|---:|
| exp(+i omega t) | 5.47723 | 0 | u | 1.18938065e-07 | 4.91282364e-07 | 6.77665675e-06 |
| exp(+i omega t) | 5.47723 | 0 | i | 8.73367232e-07 | 3.78445305e-06 | 4.33685681e-05 |
| exp(+i omega t) | 5.47723 | 0 | Zin | 8.73366852e-07 | 3.78445305e-06 | 4.33685681e-05 |
| exp(+i omega t) | 5.47723 | 0 | pressure | 2.12177667e-06 | 1.01661445e-05 | 0.000105005844 |
| exp(+i omega t) | 9.12871 | 0 | u | 1.23767891e-07 | 5.24659299e-07 | 7.08082155e-06 |
| exp(+i omega t) | 9.12871 | 0 | i | 1.32434337e-06 | 7.82533244e-06 | 5.76489359e-05 |
| exp(+i omega t) | 9.12871 | 0 | Zin | 1.32434218e-06 | 7.82533244e-06 | 5.76489359e-05 |
| exp(+i omega t) | 9.12871 | 0 | pressure | 1.64565928e-06 | 9.24286752e-06 | 9.42288767e-05 |
| exp(+i omega t) | 5.47723 | 3 | u | 1.15280499e-07 | 4.42967041e-07 | 6.59599934e-06 |
| exp(+i omega t) | 5.47723 | 3 | i | 8.49998426e-07 | 3.67717185e-06 | 4.22309833e-05 |
| exp(+i omega t) | 5.47723 | 3 | Zin | 8.49998067e-07 | 3.67717185e-06 | 4.22309833e-05 |
| exp(+i omega t) | 5.47723 | 3 | pressure | 2.12177667e-06 | 1.01661445e-05 | 0.000105005844 |
| exp(+i omega t) | 9.12871 | 3 | u | 1.20460073e-07 | 4.63718451e-07 | 6.89194e-06 |
| exp(+i omega t) | 9.12871 | 3 | i | 1.28891229e-06 | 7.60823769e-06 | 5.78836964e-05 |
| exp(+i omega t) | 9.12871 | 3 | Zin | 1.28891116e-06 | 7.60823769e-06 | 5.78836964e-05 |
| exp(+i omega t) | 9.12871 | 3 | pressure | 1.66325922e-06 | 1.00552654e-05 | 9.52477678e-05 |
| exp(-i omega t) | 5.47723 | 0 | u | 1.18938065e-07 | 4.91282364e-07 | 6.77665675e-06 |
| exp(-i omega t) | 5.47723 | 0 | i | 8.73367232e-07 | 3.78445305e-06 | 4.33685681e-05 |
| exp(-i omega t) | 5.47723 | 0 | Zin | 8.73366852e-07 | 3.78445305e-06 | 4.33685681e-05 |
| exp(-i omega t) | 5.47723 | 0 | pressure | 2.12177667e-06 | 1.01661445e-05 | 0.000105005844 |
| exp(-i omega t) | 9.12871 | 0 | u | 1.23767891e-07 | 5.24659299e-07 | 7.08082155e-06 |
| exp(-i omega t) | 9.12871 | 0 | i | 1.32434337e-06 | 7.82533244e-06 | 5.76489359e-05 |
| exp(-i omega t) | 9.12871 | 0 | Zin | 1.32434218e-06 | 7.82533244e-06 | 5.76489359e-05 |
| exp(-i omega t) | 9.12871 | 0 | pressure | 1.64565928e-06 | 9.24286752e-06 | 9.42288767e-05 |
| exp(-i omega t) | 5.47723 | 3 | u | 1.15280499e-07 | 4.42967041e-07 | 6.59599934e-06 |
| exp(-i omega t) | 5.47723 | 3 | i | 8.49998426e-07 | 3.67717185e-06 | 4.22309833e-05 |
| exp(-i omega t) | 5.47723 | 3 | Zin | 8.49998067e-07 | 3.67717185e-06 | 4.22309833e-05 |
| exp(-i omega t) | 5.47723 | 3 | pressure | 2.12177667e-06 | 1.01661445e-05 | 0.000105005844 |
| exp(-i omega t) | 9.12871 | 3 | u | 1.20460073e-07 | 4.63718451e-07 | 6.89194e-06 |
| exp(-i omega t) | 9.12871 | 3 | i | 1.28891229e-06 | 7.60823769e-06 | 5.78836964e-05 |
| exp(-i omega t) | 9.12871 | 3 | Zin | 1.28891116e-06 | 7.60823769e-06 | 5.78836964e-05 |
| exp(-i omega t) | 9.12871 | 3 | pressure | 1.66325922e-06 | 1.00552654e-05 | 9.52477678e-05 |

Decision: every measured point is below 1e-2 complex relative error. Float32 is allowed, backend precision defaults are preserved, and Metal exterior transducers are enabled with the network kept in Float64. CUDA and ROCm are unqualified. This scope does not establish accuracy for arbitrary higher-Q geometries.

Reproduce through the compute broker using `julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local scripts/measure_exterior_transducer_precision.jl`. The script prints every per-frequency metric as JSON. Public-path regression coverage is in `exterior_transducer_precision_tests.jl`; Metal hardware coverage is in `exterior_transducer_metal_tests.jl`.
