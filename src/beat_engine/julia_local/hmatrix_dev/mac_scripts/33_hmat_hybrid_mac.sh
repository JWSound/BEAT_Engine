#!/usr/bin/env bash
# H-mátrix teljes felépítés: közeli blokkok a Mac GPU-ján (Metal), tömörített blokkok a processzoron.
# 1) 11k háló + sűrű ellenőrzés, 2) 22k, 3) 43k háló, 20 kHz. Kb. 25-30 perc.
source "$(dirname "$0")/_common.sh"
ENGINE="$FORK/BEAT_Engine/src/beat_engine/julia_local"
W="$KIT/hmat/work"; rm -rf "$W"; mkdir -p "$W/src" "$W/tests"
cp "$ENGINE"/src/*.jl "$W/src/"; cp "$KIT/fmm/BeatEngineFmm.jl" "$W/src/"
cp "$KIT"/fmm/fmm_*.jl "$W/tests/"; cp "$KIT"/hmat/hmat_*.jl "$W/tests/"
export JULIA_LOAD_PATH="@:$KIT/fmm/env:@stdlib"
CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
LOG="$KIT/results/hmathybrid_$(date +%Y%m%d-%H%M).log"; mkdir -p "$KIT/results"
say "Hibrid (GPU + CPU) mérés indul, napló: $LOG"
cd "$W"
for spec in "piece_h343 1" "piece_h245 0" "piece_h17 0"; do
  set -- $spec
  echo "--- $1 20000 Hz hybrid metal $(date +%H:%M)" | tee -a "$LOG"
  HMAT_GPU=metal HMAT_DENSE=$2 /usr/bin/time -l "$JULIA" --heap-size-hint=12G --project="$ENGINE" -t "$CORES" "$W/tests/hmat_fast_test.jl" \
    "$KIT/fmm/meshes/$1" 20000 1e-3 2>&1 | grep -E "mesh |hybrid|GMRES|DENSE|FP16 plain|maximum resident|ERROR|rror" | tee -a "$LOG" || true
done
say "KÉSZ. Napló: $LOG"
