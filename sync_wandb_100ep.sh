#!/bin/bash
# Run on the login node to keep the 100ep offline wandb run synced to the cloud.
# Usage: bash sync_wandb_100ep.sh <SLURM_JOB_ID>
#   or:  bash sync_wandb_100ep.sh   (will look up the oasis-100ep job automatically)

WANDB_OFFLINE_DIR=/scratch/user/lsokoudj/PERSIST/outputs/finetune_oasis_scaling_100ep/wandb
VENV=/scratch/user/lsokoudj/PERSIST/.venv/bin/activate
INTERVAL=30  # seconds between syncs

JOB_ID="$1"
if [ -z "$JOB_ID" ]; then
    JOB_ID=$(squeue -u lsokoudj -h -o "%i" --name=oasis-100ep | head -1)
    if [ -z "$JOB_ID" ]; then
        echo "No oasis-100ep job found in queue. Submit the job first."
        exit 1
    fi
    echo "Auto-detected job ID: $JOB_ID"
fi

source "$VENV"

echo "Syncing offline wandb runs in: $WANDB_OFFLINE_DIR"
echo "Monitoring job $JOB_ID — will sync every ${INTERVAL}s until it finishes."
echo "Press Ctrl+C to stop early (run will remain in offline state until next sync)."
echo ""

sync_once() {
    local run_dirs
    run_dirs=$(find "$WANDB_OFFLINE_DIR" -maxdepth 1 -name 'offline-run-*' -type d 2>/dev/null)
    if [ -z "$run_dirs" ]; then
        echo "$(date '+%H:%M:%S') No offline run dirs yet, waiting..."
        return
    fi
    for d in $run_dirs; do
        echo "$(date '+%H:%M:%S') Syncing $d ..."
        wandb sync "$d" 2>&1 | grep -v "^wandb: $" | grep -v "^$"
    done
}

while true; do
    job_state=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
    if [ -z "$job_state" ]; then
        echo "$(date '+%H:%M:%S') Job $JOB_ID finished. Doing final sync..."
        sync_once
        echo "Done. View run at: https://wandb.ai/leopoldgatsing-texas-a-m-university/persist-oasis-finetune"
        break
    fi

    sync_once
    sleep "$INTERVAL"
done
