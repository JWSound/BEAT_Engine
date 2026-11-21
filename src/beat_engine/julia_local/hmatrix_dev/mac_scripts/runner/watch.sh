#!/bin/bash
# Claude job runner: runs whitelisted kit scripts dropped into _beat_kit/queue/ as <NN_name.sh>.job files.
# Only scripts that exist directly in _beat_kit and match NN_name.sh are allowed. One job at a time.
KIT="$HOME/BoundaryLab/_beat_kit"; Q="$KIT/queue"; mkdir -p "$Q/done" "$KIT/results"
LOG="$KIT/results/runner.log"
echo "$(date '+%F %T') runner started (pid $$)" >> "$LOG"
while true; do
  for job in "$Q"/*.job; do
    [ -e "$job" ] || continue
    name=$(basename "$job" .job)
    if [[ "$name" =~ ^[0-9]{2}_[a-z0-9_]+\.sh$ ]] && [ -f "$KIT/$name" ]; then
      mv "$job" "$Q/done/$(date +%Y%m%d-%H%M%S)_$name.job"
      echo "$(date '+%F %T') START $name" >> "$LOG"
      bash "$KIT/$name" >> "$KIT/results/runner_$(date +%Y%m%d-%H%M)_${name%.sh}.out" 2>&1
      echo "$(date '+%F %T') END $name (exit $?)" >> "$LOG"
    else
      mv "$job" "$Q/done/REJECTED_$(basename "$job")"
      echo "$(date '+%F %T') REJECTED $name" >> "$LOG"
    fi
  done
  sleep 15
done
