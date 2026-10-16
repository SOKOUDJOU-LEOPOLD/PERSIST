#!/bin/bash
# Run on the login node (compute nodes have no internet): uploads the offline wandb runs of the
# full29k trainings every INTERVAL seconds until no train-* job is left in the queue.
# Usage: nohup bash dataset_toolkits/manifests/full29k/sync_wandb_full29k.sh > outputs/runs_full29k/logs/wandb_sync.log 2>&1 &

WANDB_ROOT=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/wandb
INTERVAL=300

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
while true; do
    for run in "$WANDB_ROOT"/offline-run-*; do
        [ -d "$run" ] && wandb sync --include-offline "$run" > /dev/null 2>&1 || echo "$(date) sync failed: $run"
    done
    echo "$(date) synced $(ls -d "$WANDB_ROOT"/offline-run-* 2>/dev/null | wc -l) run(s)"
    squeue -u "$USER" -h -o "%j" | grep -q "^train-" || { echo "$(date) no training jobs left, final sync done"; break; }
    sleep $INTERVAL
done
