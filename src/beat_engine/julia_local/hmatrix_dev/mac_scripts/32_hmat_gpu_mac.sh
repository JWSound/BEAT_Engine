#!/usr/bin/env bash
# GPU (Metal) magfüggvény teszt: a H-mátrix közeli blokkjainak számítása a Mac GPU-ján, egyszeres pontossággal.
# Első futáskor telepíti a Metal.jl-t (~3-5 perc). A gép közben használható.
source "$(dirname "$0")/_common.sh"
ENGINE="$FORK/BEAT_Engine/src/beat_engine/julia_local"
[ -d "$ENGINE" ] || die "nem találom a forkot: $ENGINE"
W="$KIT/hmat/gpuwork"; rm -rf "$W"; mkdir -p "$W/src" "$W/tests"
cp "$ENGINE"/src/*.jl "$W/src/"; cp "$KIT/fmm/BeatEngineFmm.jl" "$W/src/"
cp "$KIT"/fmm/fmm_*.jl "$W/tests/"; cp "$KIT"/hmat/hmat_*.jl "$W/tests/"
ENV_DIR="$KIT/fmm/env"
say "Metal / KernelAbstractions telepítése (egyszer, pár perc)"
JULIA_PROJECT="$ENV_DIR" "$JULIA" -e 'using Pkg; Pkg.add(["Metal","KernelAbstractions","Adapt"]); Pkg.precompile(); using Metal; Metal.versioninfo()' 2>&1 | tail -15
export JULIA_LOAD_PATH="@:$ENV_DIR:@stdlib"
LOG="$KIT/results/hmatgpu_$(date +%Y%m%d-%H%M).log"; mkdir -p "$KIT/results"
say "GPU teszt indul, napló: $LOG"
cd "$W"
for mesh in piece_h343 piece_h245; do
  echo "--- $mesh 20000 Hz metal $(date +%H:%M)" | tee -a "$LOG"
  "$JULIA" --project="$ENGINE" -t 4 "$W/tests/hmat_ka_test.jl" "$KIT/fmm/meshes/$mesh" 20000 metal 2>&1 \
    | grep -E "mesh |kernel|accuracy|ERROR|error" | tee -a "$LOG" || true
done
say "KÉSZ. Napló: $LOG"
