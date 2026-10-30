#!/bin/bash
# Download the gated dynamic dataset Subarna2026/seg10k (202 GB, 80k files) to datasets/seg10k on Grace.
# Hugging Face rate-limits bursts (HTTP 429), so few workers and a retry loop: snapshot_download skips files that
# are already complete, so each retry resumes where the last one stopped.
# Usage: nohup setsid bash dataset_toolkits/download_seg10k.sh > outputs/logs/seg10k_download.log 2>&1 < /dev/null &
cd /scratch/user/lsokoudj/PERSIST
source .venv/bin/activate
export HF_HOME=/scratch/user/lsokoudj/.cache/huggingface        # keep caches off the 10 GB home quota
export HF_TOKEN_PATH=/home/lsokoudj/.cache/huggingface/token    # dataset is gated
for attempt in $(seq 1 50); do
    echo "=== attempt $attempt, $(date) ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Subarna2026/seg10k', repo_type='dataset', local_dir='datasets/seg10k', max_workers=4)
" && { echo "DOWNLOAD COMPLETE $(date)"; exit 0; }
    sleep 300
done
echo "GAVE UP after $attempt attempts"; exit 1
