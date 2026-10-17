#!/bin/bash
# Run on the login node (compute nodes have no internet): uploads the offline wandb runs of the
# full29k trainings every INTERVAL seconds until no train-* job is left in the queue.
# Usage: nohup bash dataset_toolkits/manifests/full29k/sync_wandb_full29k.sh > outputs/runs_full29k/logs/wandb_sync.log 2>&1 &
#
# Uses `wandb beta sync`: the classic `wandb sync` in wandb 0.25.0 crashes in upsert_run
# (server_settings_introspection gets an empty reply) and still marks the run as synced.
# --no-skip-synced re-uploads runs that are still being written; re-syncing a run is idempotent.

WANDB_ROOT=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/wandb
INTERVAL=300

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
while true; do
    if ls -d "$WANDB_ROOT"/offline-run-* > /dev/null 2>&1; then
        wandb beta sync --no-skip-synced "$WANDB_ROOT" 2>&1 | grep -E "Finished syncing|rror" | sed "s/^/$(date '+%F %T') /"
    fi
    squeue -u "$USER" -h -o "%j" | grep -q "^train-" || { echo "$(date) no training jobs left, final sync done"; break; }
    sleep $INTERVAL
done
