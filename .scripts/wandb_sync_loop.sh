#!/bin/bash
source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate

# One-time sync for runs that are already finished (no new data expected).
DONE_DIRS=(
  "/scratch/user/lsokoudj/PERSIST/baselines"
  "/scratch/user/lsokoudj/PERSIST/outputs/finetune_oasis_scaling_100ep"
  "/scratch/user/lsokoudj/PERSIST/outputs/finetune_oasis_scaling_206ep"
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
ACTIVE_DIRS=(
  "/scratch/user/lsokoudj/PERSIST/outputs/finetune_oasis_luanti_1000ep"
  "/scratch/user/lsokoudj/PERSIST/outputs/finetune_oasis_luanti_hudnav_1000ep"
)

while true; do
  for d in "${ACTIVE_DIRS[@]}"; do
    RUNDIR="$d/wandb/latest-run"
    if [ -e "$RUNDIR" ]; then
      echo "[$(date)] append-syncing $(readlink -f "$RUNDIR")"
      wandb sync --append "$RUNDIR" 2>&1
    else
      echo "[$(date)] no latest-run symlink found at $RUNDIR"
    fi
  done
  echo "[$(date)] sleeping 5m"
  sleep 300
done
