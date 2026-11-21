#!/usr/bin/env bash
# H-mátrix pontosság ellenőrzése sűrű megoldással: 11k és 22k háló, 20 kHz.
# Kimenet: tárolás, max rang, felépítési idő, GMRES iteráció, FP16/BF16 hiba, csúcsmemória.
# A gép közben használható, de nagy programot ne futtass mellette.
source "$(dirname "$0")/_common.sh"
ENGINE="$FORK/BEAT_Engine/src/beat_engine/julia_local"
[ -d "$ENGINE" ] || die "nem találom a forkot: $ENGINE"
W="$KIT/hmat/work"; rm -rf "$W"; mkdir -p "$W/src" "$W/tests"
cp "$ENGINE"/src/*.jl "$W/src/"; cp "$KIT/fmm/BeatEngineFmm.jl" "$W/src/"
cp "$KIT"/fmm/fmm_*.jl "$W/tests/"; cp "$KIT"/hmat/hmat_*.jl "$W/tests/"
ENV_DIR="$KIT/fmm/env"
export JULIA_LOAD_PATH="@:$ENV_DIR:@stdlib"
CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
LOG="$KIT/results/hmatdense_$(date +%Y%m%d-%H%M).log"; mkdir -p "$KIT/results"
say "H-mátrix mérés indul, napló: $LOG  (magok: $CORES)"
cd "$W"
for mesh in piece_h343 piece_h245; do
  echo "--- $mesh 20000 Hz $(date +%H:%M)" | tee -a "$LOG"
  HMAT_DENSE=1 /usr/bin/time -l "$JULIA" --heap-size-hint=12G --project="$ENGINE" -t "$CORES" "$W/tests/hmat_fast_test.jl" \
    "$KIT/fmm/meshes/$mesh" 20000 1e-3 2>&1 | grep -E "mesh |GMRES|DENSE|FP16 plain|BF16|Jacobi|cluster|maximum resident|ERROR" | tee -a "$LOG" || true
done
say "KÉSZ. Napló: $LOG"
