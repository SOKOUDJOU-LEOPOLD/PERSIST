#!/bin/bash
# Push datasets/luanti_full29k to csce-chzhang.engr.tamu.edu (resumable: rerun this script to continue).
# Usage: nohup bash dataset_toolkits/manifests/full29k/transfer_to_csce.sh > outputs/transfer_csce.log 2>&1 &
set -uo pipefail
SRC=/scratch/user/lsokoudj/PERSIST/datasets/luanti_full29k
DST=lsokoudj@csce-chzhang.engr.tamu.edu
DST_DS=/home/lsokoudj/PERSIST/datasets/luanti_full29k
DST_RAW=/data/lsokoudj/luanti_full29k_raw
DST_TMP=/data/lsokoudj/luanti_full29k_small_files
# npz/mp4 are already compressed: no -z
RSYNC="rsync -a --partial --info=stats1,progress2 -e ssh"

step() { echo "=== $(date '+%F %T') $* ==="; }

step "metadata, config, small-file tars"
ssh $DST "mkdir -p $DST_DS $DST_RAW $DST_TMP"
$RSYNC $SRC/metadata.csv $SRC/dataset_params.json $SRC/mt_voxel_classdict.json $SRC/latent_stats.npz $DST:$DST_DS/ || exit 1
$RSYNC ~/persist_small_files/small_files_*.tar $DST:$DST_TMP/ || exit 1

step "voxel_classes/ (165.5 GB, 29,008 files)"
$RSYNC $SRC/voxel_classes $DST:$DST_DS/ || exit 1

step "raw/ (327.7 GB, 58,016 files)"
$RSYNC $SRC/raw/ $DST:$DST_RAW/ || exit 1

step "restore sha256.txt / level_metadata.json into raw/, verify counts"
ssh $DST "for t in $DST_TMP/small_files_*.tar; do tar -xf \$t -C $DST_RAW || exit 1; done
  echo voxel_classes: \$(ls $DST_DS/voxel_classes | wc -l) '(expect 29008)'
  echo s1nav episodes: \$(ls $DST_RAW/OpenWorldCreative-v0 | wc -l) '(expect 9463)'
  echo hudnav episodes: \$(ls $DST_RAW/OpenWorldCreativeHud-v0 | wc -l) '(expect 19545)'
  echo files under raw: \$(find $DST_RAW -type f | wc -l) '(expect 116032)'
  md5sum $DST_DS/mt_voxel_classdict.json"
step "DONE"
