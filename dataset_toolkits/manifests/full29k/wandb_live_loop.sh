#!/bin/bash
# Run on the login node: keeps a running offline wandb run's charts current. `wandb beta sync --live` stops
# picking up new history after a while, so it is restarted every 15 min (re-uploading is idempotent) until the
# Slurm job ends, then one final sync. The plain (non-live) sync can't read a run that is still being written.
# Usage: nohup bash wandb_live_loop.sh <offline-run dir> <slurm job name> > <log> 2>&1 &
cd /scratch/user/lsokoudj/PERSIST
source .venv/bin/activate
while squeue -u lsokoudj -h -o %j | grep -qx "$2"; do
    timeout 900 wandb beta sync --live --no-skip-synced "$1" 2>&1 | grep -o "uploading history steps [0-9-]*" | tail -1 | sed "s/^/$(date '+%F %T') /"
done
wandb beta sync --no-skip-synced "$1" 2>&1 | tail -1
echo "$(date) job $2 gone, final sync done"
