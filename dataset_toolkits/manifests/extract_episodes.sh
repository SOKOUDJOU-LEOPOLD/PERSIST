#!/bin/bash
set -uo pipefail
cd /home/lsokoudj/PERSIST
TOK=$(cat ~/.cache/huggingface/token)
mkdir -p datasets_staging/raw_extract
cd datasets_staging/raw_extract

echo "=== s1nav extraction starting $(date) ==="
mapfile -t SEEDS < /home/lsokoudj/PERSIST/dataset_toolkits/manifests/s1nav_seeds.txt
PATTERNS=()
for s in "${SEEDS[@]}"; do
    PATTERNS+=("OpenWorldCreative-v0/${s}" "OpenWorldCreative-v0/${s}/*")
done
curl -sL -H "Authorization: Bearer $TOK" "https://huggingface.co/datasets/percool/luanti/resolve/main/motworld-s1nav-10k.tar.zst" \
  | zstd -dc | tar -xf - --wildcards "${PATTERNS[@]}"
echo "=== s1nav extraction done $(date), got $(ls OpenWorldCreative-v0/ 2>/dev/null | wc -l) dirs ==="

echo "=== hudnav extraction starting $(date) ==="
mapfile -t SEEDS < /home/lsokoudj/PERSIST/dataset_toolkits/manifests/hudnav_seeds.txt
PATTERNS=()
for s in "${SEEDS[@]}"; do
    PATTERNS+=("OpenWorldCreativeHud-v0/${s}" "OpenWorldCreativeHud-v0/${s}/*")
done
curl -sL -H "Authorization: Bearer $TOK" "https://huggingface.co/datasets/percool/luanti/resolve/main/motworld-hudnav-20k.tar.zst" \
  | zstd -dc | tar -xf - --wildcards "${PATTERNS[@]}"
echo "=== hudnav extraction done $(date), got $(ls OpenWorldCreativeHud-v0/ 2>/dev/null | wc -l) dirs ==="

echo "=== ALL EXTRACTION COMPLETE $(date) ==="
