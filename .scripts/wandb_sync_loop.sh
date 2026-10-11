#!/bin/bash
# Run on a LOGIN node (compute nodes have no internet, so training logs offline):
#   nohup .scripts/wandb_sync_loop.sh > outputs/wandb_sync_loop.log 2>&1 &
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/.venv/bin/activate"

# One-time sync for runs that are already finished (no new data expected).
DONE_DIRS=(
  "$REPO/baselines"
  "$REPO/outputs/finetune_oasis_scaling_100ep"
  "$REPO/outputs/finetune_oasis_scaling_206ep"
)
for d in "${DONE_DIRS[@]}"; do
  if [ -d "$d/wandb" ]; then
    echo "[$(date)] one-time sync: $d"
    (cd "$d" && wandb sync --sync-all 2>&1)
  fi
done

# Continuously append-sync the active runs, which are still being written to.
# Plain `wandb sync` closes out the run each call, so new data after the first
# sync gets skipped; --append keeps pushing new steps without finalizing.
# Every resumed slurm job writes a NEW offline-run-*-<same id> dir, and latest-run moves to it, so
# each older dir gets one last append-sync (for the steps logged after the previous pass) and is then
# marked with .final_synced so it is not re-sent every loop.
ACTIVE_DIRS=(
  "$REPO/outputs/finetune_oasis_luanti_1000ep"
  "$REPO/outputs/finetune_oasis_luanti_hudnav_1000ep"
)

while true; do
  for d in "${ACTIVE_DIRS[@]}"; do
    LATEST=$(readlink -f "$d/wandb/latest-run" 2>/dev/null)
    if [ -z "$LATEST" ]; then
      echo "[$(date)] no latest-run symlink found at $d/wandb/latest-run"
      continue
    fi
    for RUNDIR in "$d"/wandb/offline-run-*; do
      [ -d "$RUNDIR" ] || continue
      [ -e "$RUNDIR/.final_synced" ] && continue
      echo "[$(date)] append-syncing $RUNDIR"
      wandb sync --append "$RUNDIR" 2>&1
      if [ "$(readlink -f "$RUNDIR")" != "$LATEST" ]; then
        touch "$RUNDIR/.final_synced"
      fi
    done
  done
  echo "[$(date)] sleeping 5m"
  sleep 300
done
