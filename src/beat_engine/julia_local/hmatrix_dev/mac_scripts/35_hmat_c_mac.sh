#!/usr/bin/env bash
# A második BEM-operátor (-C, a csatolt rendszer jobb oldala) H-mátrixa a Mac GPU-ján: 11k / 22k / 43k háló, 20 kHz.
source "$(dirname "$0")/_common.sh"
ENGINE="$FORK/BEAT_Engine/src/beat_engine/julia_local"
W="$KIT/hmat/cwork"; rm -rf "$W"; mkdir -p "$W/src" "$W/tests"
cp "$ENGINE"/src/*.jl "$W/src/"; cp "$KIT/fmm/BeatEngineFmm.jl" "$W/src/"
cp "$KIT"/fmm/fmm_*.jl "$W/tests/"; cp "$KIT"/hmat/hmat_*.jl "$W/tests/"
export JULIA_LOAD_PATH="@:$KIT/fmm/env:@stdlib"
CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
LOG="$KIT/results/hmatc_$(date +%Y%m%d-%H%M).log"; mkdir -p "$KIT/results"
cd "$W"
for mesh in piece_h343 piece_h245 piece_h17; do
  echo "--- $mesh 20000 Hz C metal $(date +%H:%M)" | tee -a "$LOG"
  /usr/bin/time -l "$JULIA" --heap-size-hint=12G --project="$ENGINE" -t "$CORES" "$W/tests/hmat_c_test.jl" \
    "$KIT/fmm/meshes/$mesh" 20000 1e-3 metal 2>&1 | grep -E "C build|mesh |driver|random|maximum resident|ERROR|rror" | tee -a "$LOG" || true
done
echo "DONE $(date +%H:%M)" | tee -a "$LOG"
