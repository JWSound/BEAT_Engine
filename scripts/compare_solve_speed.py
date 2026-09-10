"""Compare a solve-speed benchmark run against a committed baseline.

This is the regression gate for step 1 of the performance plan. It has two
independent checks:

  * correctness (always on) -- the per-frequency pressure and field norms are
    Float32-deterministic for the CPU path, so any real change to the numerics
    drifts them. A drift beyond ``--accuracy-tolerance`` fails, because a speed
    fix that silently moves the answer is not a fix.

  * speed (opt-in) -- each stage's median sweep time must not exceed the
    baseline's median by more than ``--time-threshold``. This is off by default
    because wall-clock time is only meaningful on a stable host: shared CI
    runners are too noisy for a tight threshold, so CI passes ``--time-threshold``
    at a very wide value (or omits it and gates correctness only), and the tight
    number is applied on the self-hosted hardware runner / a developer machine.

The baseline and the results must describe the same workload (mesh, dof count,
frequency range, backend, precision) or the comparison is refused rather than
reported as a pass.

Usage:
    python scripts/compare_solve_speed.py \
        --baseline results/baseline_sweep_cpu.json \
        --results   results/sweep.json \
        [--time-threshold 1.25] [--accuracy-tolerance 1e-4]
"""

import argparse
import json
import sys

STAGE_KEYS = (
    "operator_assembly",
    "system_build",
    "linear_solve",
    "solve_total",
    "field_evaluation",
)

WORKLOAD_KEYS = (
    "mesh",
    "p1_dofs",
    "dp0_dofs",
    "min_freq",
    "max_freq",
    "steps",
    "backend",
    "precision",
    "eval_points",
)


def relative_diff(a, b):
    if b == 0:
        return abs(a)
    return abs(a - b) / abs(b)


def check_workload(baseline, results):
    mismatches = []
    for key in WORKLOAD_KEYS:
        base_val = baseline.get(key)
        res_val = results.get(key)
        if base_val != res_val:
            mismatches.append(f"{key}: baseline {base_val!r} != results {res_val!r}")
    return mismatches


def correctness_failures(baseline, results, tolerance):
    base_freqs = baseline.get("frequencies", [])
    res_freqs = results.get("frequencies", [])
    if len(base_freqs) != len(res_freqs):
        return [f"frequency count changed: baseline {len(base_freqs)} vs results {len(res_freqs)}"]

    failures = []
    for i, (bf, rf) in enumerate(zip(base_freqs, res_freqs)):
        freq = bf.get("frequency_hz", rf.get("frequency_hz"))
        for marker in ("pressure_norm", "field_norm"):
            base_val = bf.get(marker)
            res_val = rf.get(marker)
            if base_val is None or res_val is None:
                # A marker absent from either side is not compared (e.g. a run
                # with --eval-points 0 has no field norm).
                continue
            drift = relative_diff(res_val, base_val)
            if drift > tolerance:
                failures.append(
                    f"{marker} @ {freq:.1f} Hz: baseline {base_val:.8g} vs "
                    f"results {res_val:.8g} (relative {drift:.3e} > {tolerance:.1e})"
                )
    return failures


def speed_failures(baseline, results, threshold):
    base_summary = baseline.get("summary_seconds", {})
    res_summary = results.get("summary_seconds", {})
    failures, notes = [], []
    for stage in STAGE_KEYS:
        res_med = res_summary.get(stage, {}).get("median")
        if res_med is None:
            continue
        # Finiteness is always checked: a hung or NaN'd stage must fail even on
        # a host where no time threshold is being enforced.
        if not (res_med >= 0):
            failures.append(f"{stage}: non-finite/negative timing {res_med!r}")
            continue
        base_med = base_summary.get(stage, {}).get("median")
        if base_med is None or base_med <= 0:
            continue
        ratio = res_med / base_med
        if threshold is not None and ratio > threshold:
            failures.append(
                f"{stage}: median {res_med:.4f}s vs baseline {base_med:.4f}s "
                f"({ratio:.2f}x > {threshold:.2f}x)"
            )
        elif threshold is not None:
            notes.append(f"{stage}: {res_med:.4f}s vs {base_med:.4f}s ({ratio:.2f}x)")
    return failures, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="committed baseline JSON")
    parser.add_argument("--results", required=True, help="benchmark run JSON to check")
    parser.add_argument("--time-threshold", type=float, default=None,
                        help="fail if any stage median exceeds the baseline by more than this factor")
    parser.add_argument("--accuracy-tolerance", type=float, default=1e-4,
                        help="fail if a correctness norm drifts by more than this relative amount")
    args = parser.parse_args(argv)

    with open(args.baseline, encoding="utf-8") as fh:
        baseline = json.load(fh)
    with open(args.results, encoding="utf-8") as fh:
        results = json.load(fh)

    workload_mismatches = check_workload(baseline, results)
    if workload_mismatches:
        print("FAIL: baseline and results are not the same workload; refusing to compare.")
        for m in workload_mismatches:
            print("  - " + m)
        print("Re-record the baseline with the same --mesh / --steps / --min-freq / --max-freq / --backend.")
        return 2

    accuracy_failures = correctness_failures(baseline, results, args.accuracy_tolerance)
    time_failures, time_notes = speed_failures(baseline, results, args.time_threshold)

    stage = baseline.get("summary_seconds", {}).get("total_sweep", {})
    print("Comparing solve-speed run:")
    print(f"  workload: {results.get('mesh')} | {results.get('backend')} | "
          f"{results.get('precision')} | {results.get('p1_dofs')} P1 dofs | "
          f"{results.get('steps')} steps {results.get('min_freq')}-{results.get('max_freq')} Hz")
    print(f"  baseline total_sweep median: {stage.get('median', 'n/a'):.4f} s  "
          f"(accuracy tolerance {args.accuracy_tolerance:.1e})")

    for note in time_notes:
        print("  " + note)

    failures = accuracy_failures + time_failures
    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  - " + f)
        return 1

    print("\nPASS: correctness within tolerance"
          + (f" and no stage regressed past {args.time_threshold:.2f}x" if args.time_threshold else "")
          + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())