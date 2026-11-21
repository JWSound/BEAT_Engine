#!/usr/bin/env bash
# Mint a 36-os, de a sűrű referencia a CPU-n, dupla pontosságú összerakással készül (beat_cpu).
# 2) A H-mátrixos megoldó (GPU) ugyanezt megoldja, és összeveti a sűrű megoldással (BEM-nyomás, fluxus, mechanikai, elektromos).
source "$(dirname "$0")/_common.sh"
B="$FORK/BEAT_Engine"; F="src/beat_engine/julia_local/src/BeatEngineCoupledCondensed.jl"
PATCH="$KIT/patches/hmat_dump_hook.diff"
git -C "$B" apply --check "$PATCH" || die "a fejlesztői kapcsoló nem illeszthető a forkra"
git -C "$B" apply "$PATCH"
revert() { git -C "$B" apply -R "$PATCH" && echo "fejlesztői kapcsoló visszavonva"; }
trap revert EXIT
RES="$KIT/results/$(date +%Y%m%d-%H%M)-hmat-real-cpu"; mkdir -p "$RES"; cd "$RES"
DUMP="$RES/dump"
P="$KIT/reference/20261120-conv_piece_h343/project/HF_piece_h343.blab.json"
[ -f "$P" ] || P=$(ls "$KIT"/reference/20261120-conv_piece_h343/project/*.blab.json | head -1)
echo "--- dense solve with dump $(date +%H:%M)"
"$FORK/venv/bin/python" "$KIT/tools/fork_compare.py" run h343_20k_dump_cpu "$FORK/venv" "$P" "$KIT/requests/f20k.json" beat_cpu \
  BLAB_HMAT_DUMP="$DUMP" BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION=off BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION=off \
  BLAB_COUPLED_DENSE_SPLIT=off BLAB_COUPLED_INTERFACE_RADIATION=off
tail -5 logs/solve_h343_20k_dump_cpu.log
revert; trap - EXIT
ls -la "$DUMP"
ENGINE="$B/src/beat_engine/julia_local"
W="$KIT/hmat/realwork"; rm -rf "$W"; mkdir -p "$W/src" "$W/tests"
cp "$ENGINE"/src/*.jl "$W/src/"; cp "$KIT/fmm/BeatEngineFmm.jl" "$W/src/"; cp "$KIT"/fmm/fmm_*.jl "$W/tests/"; cp "$KIT"/hmat/hmat_*.jl "$W/tests/"
export JULIA_LOAD_PATH="@:$KIT/fmm/env:@stdlib"
CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
cd "$W"
for f in "$DUMP"/*/; do
  for tol in 1e-4; do
    echo "--- H check $(basename $f) tol $tol $(date +%H:%M)"
    "$JULIA" --heap-size-hint=12G --project="$ENGINE" -t "$CORES" "$W/tests/hmat_dump_check.jl" "$f" metal $tol 2>&1 | grep -vE "^\s*@|^ \[|Warning|└|┌"
  done
done
echo "DONE $(date +%H:%M)"
