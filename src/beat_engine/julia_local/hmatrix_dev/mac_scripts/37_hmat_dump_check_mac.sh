#!/usr/bin/env bash
# H-mátrixos ellenőrzés a legutóbbi valódi-modell dumpon (36-os futás eredménye), tol 1e-3 és 1e-4.
source "$(dirname "$0")/_common.sh"
B="$FORK/BEAT_Engine"; ENGINE="$B/src/beat_engine/julia_local"
DUMP=${HMAT_DUMP_DIR:-$(ls -td "$KIT"/results/*-hmat-real/dump | head -1)}; [ -d "$DUMP" ] || die "nincs dump"
W="$KIT/hmat/realwork"; rm -rf "$W"; mkdir -p "$W/src" "$W/tests"
cp "$ENGINE"/src/*.jl "$W/src/"; cp "$KIT/fmm/BeatEngineFmm.jl" "$W/src/"; cp "$KIT"/fmm/fmm_*.jl "$W/tests/"; cp "$KIT"/hmat/hmat_*.jl "$W/tests/"
export JULIA_LOAD_PATH="@:$KIT/fmm/env:@stdlib"
CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
cd "$W"
for f in "$DUMP"/*/; do
  for tol in 1e-3; do
    echo "--- H check $f tol $tol $(date +%H:%M)"
    "$JULIA" --heap-size-hint=12G --project="$ENGINE" -t "$CORES" "$W/tests/hmat_dump_check.jl" "$f" metal $tol 2>&1 | grep -vE "^\s*@|^ \[|Warning|└|┌"
  done
done
echo "DONE $(date +%H:%M)"
