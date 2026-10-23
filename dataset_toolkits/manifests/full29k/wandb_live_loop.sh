#!/bin/bash
# Run on the login node: keeps a running offline wandb run's charts current. `wandb beta sync --live` stops
# picking up new history after a while, so it is restarted every 15 min (re-uploading is idempotent) until the
# Slurm job ends, then one final sync. The plain (non-live) sync can't read a run that is still being written.
# Optional WANDB_ID uploads into an existing wandb run (e.g. append a resumed continuation to the original run).
# Usage: nohup bash wandb_live_loop.sh '<offline-run dir or glob>' <slurm job name> [WANDB_ID] > <log> 2>&1 &
cd /scratch/user/lsokoudj/PERSIST
source .venv/bin/activate
ID=${3:+--id $3}
running() { squeue -u lsokoudj -h -o %j | grep -qx "$1"; }

RUN_DIR=""
while running "$2" && [ -z "$RUN_DIR" ]; do  # the offline-run dir appears only once training has started
    RUN_DIR=$(ls -dt $1 2>/dev/null | head -1)
    [ -z "$RUN_DIR" ] && sleep 60
done
[ -z "$RUN_DIR" ] && { echo "$(date) job $2 gone before its wandb dir appeared"; exit 1; }
echo "$(date) syncing $RUN_DIR ${3:+into run $3}"

while running "$2"; do
    timeout 900 wandb beta sync --live --no-skip-synced $ID "$RUN_DIR" 2>&1 | grep -oE "uploading history steps [0-9-]*|ERROR.*" | tail -1 | sed "s/^/$(date '+%F %T') /"
done
wandb beta sync --no-skip-synced $ID "$RUN_DIR" 2>&1 | tail -1
echo "$(date) job $2 gone, final sync done"
