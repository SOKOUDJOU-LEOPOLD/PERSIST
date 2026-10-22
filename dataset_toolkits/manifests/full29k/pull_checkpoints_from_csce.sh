#!/bin/bash
# Run on the Grace login node: every INTERVAL seconds, copy the checkpoints of a training run on csce-chzhang
# (which keeps only latest_and_best for lack of disk) to Grace scratch, so every checkpoint_<step> survives.
# Never deletes anything on either side. Stops after the run's training process has exited (one last copy).
# Usage: nohup bash pull_checkpoints_from_csce.sh <component dir, e.g. camera_transformer> <run-name prefix> \
#          > outputs/runs_full29k/logs/pull_<name>.log 2>&1 &

COMPONENT_DIR=$1
RUN_PREFIX=$2
INTERVAL=${INTERVAL:-600}
HOST=lsokoudj@csce-chzhang.engr.tamu.edu
SRC=PERSIST/outputs/runs_full29k/$COMPONENT_DIR
DST=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/$COMPONENT_DIR
mkdir -p $DST

while true; do
    running=$(ssh -o BatchMode=yes $HOST "pgrep -u lsokoudj -f '[r]un-name $RUN_PREFIX' | wc -l" 2>/dev/null | tail -1)
    rsync -a --partial --include="$RUN_PREFIX*/" --include="$RUN_PREFIX*/checkpoints/***" --exclude="*" \
        $HOST:$SRC/ $DST/ 2>&1 | grep -v -E "official State|^Texas A&M|^$"
    echo "$(date '+%F %T') copied; on Grace: $(ls -d $DST/$RUN_PREFIX*/checkpoints/checkpoint_* 2>/dev/null | xargs -n1 basename | sort -t_ -k2 -n | tr '\n' ' ')"
    [ "$running" = 0 ] && { echo "$(date) training process gone, final copy done"; break; }
    sleep $INTERVAL
done
