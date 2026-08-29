"""Merge a second episode shard's outputs into a main eval_qualitative.py output directory.

Why this exists: to parallelize a dense (--voxel-render-stride 1) 6-episode re-run, each 3-episode
subset was launched as its own scripts.eval_qualitative process (via --eval-instances), each into
its own output directory. Inside a single process, episode indices always start at 0
(enumerate(dataloader)), so a "shard B" covering episodes 3-5 writes files named episode_00.mp4,
episode_01.mp4, episode_02.mp4, episode_00_trajectory.png, ... and a summary.json with
episode_index 0,1,2 -- all colliding with shard A's filenames if written to the same directory.
This script renames shard B's files with the correct offset and merges its summary.json entries
into the main summary.json, so the result looks exactly like a single 6-episode run would have.

Example:
    python -m scripts.merge_episode_shards \
        --main-dir outputs/frame_analysis_persist \
        --shard-dir outputs/frame_analysis_persist_shardB \
        --index-offset 3
"""
import json
import os
import shutil
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    main_dir: str
    shard_dir: str
    index_offset: int
    delete_shard_dir: bool = True


def main(args: Args):
    with open(os.path.join(args.shard_dir, "summary.json")) as f:
        shard_summary = json.load(f)

    with open(os.path.join(args.main_dir, "summary.json")) as f:
        main_summary = json.load(f)

    for entry in shard_summary:
        old_idx = entry["episode_index"]
        new_idx = old_idx + args.index_offset
        for suffix, ext in [("", ".mp4"), ("_trajectory", ".png")]:
            src = os.path.join(args.shard_dir, f"episode_{old_idx:02d}{suffix}{ext}")
            dst = os.path.join(args.main_dir, f"episode_{new_idx:02d}{suffix}{ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {src} -> {dst}")
            else:
                print(f"WARNING: expected shard file missing: {src}")
        entry["episode_index"] = new_idx
        main_summary.append(entry)

    main_summary.sort(key=lambda e: e["episode_index"])
    with open(os.path.join(args.main_dir, "summary.json"), "w") as f:
        json.dump(main_summary, f, indent=2)
    print(f"Merged {len(shard_summary)} episodes into {args.main_dir}/summary.json "
          f"({len(main_summary)} total episodes)")

    if args.delete_shard_dir:
        shutil.rmtree(args.shard_dir)
        print(f"Removed {args.shard_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
