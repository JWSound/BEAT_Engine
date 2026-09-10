# Sweep benchmark baselines

`baseline_sweep_cpu.json` is the committed solve-speed baseline for the step-1
regression gate. It was recorded with:

```
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local \
    src/beat_engine/julia_local/scripts/benchmark_sweep.jl \
    --mesh src/beat_engine/julia_local/test_meshes/sample.msh \
    --steps 8 --min-freq 100 --max-freq 20000 --eval-points 74 \
    --warmups 1 --repetitions 3 \
    --json src/beat_engine/julia_local/results/baseline_sweep_cpu.json
```

`scripts/compare_solve_speed.py` diffs a fresh run against this file. Its
correctness check (per-frequency pressure and field norms) is portable across
hosts; its time check is only meaningful on a stable host, so tight time
thresholds belong on the self-hosted accelerator runner and on developers'
machines, not on shared CI runners.

Re-record the baseline whenever a change intentionally moves the numerics (for
example, a coupling or quadrature change that shifts the far field within the
accuracy budget): the baseline tracks the current correct answer, not the
original one.