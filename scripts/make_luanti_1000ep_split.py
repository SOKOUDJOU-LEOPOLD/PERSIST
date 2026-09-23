"""Builds a 1000-episode training split from the Luanti motworld-s1nav-10k dataset (HuggingFace
percool/luanti), for fine-tuning Oasis from its original pretrained weights at this larger data
scale. Unlike scripts/make_oasis_scaling_subsets.py (which draws its subsets from a pre-built 3-way
finetune_split_oasis_scaling.json), this script's source is the raw HF splits.json
(data/luanti/motworld-s1nav-10k.splits.json), which itself only labels each episode "train"
(9,463), "eval" (200), or "unused" (324) -- a different dataset, a different source split, not
compatible with that earlier file's input format.

- train: --num-train episodes (default 1000), sampled from the HF "train"-labeled pool (fixed
  shuffle, seed 0).
- train_eval: a fixed 25-episode prefix of that same subset (same convention as
  make_oasis_scaling_subsets.py) -- used by finetune_oasis.py's overfitting-onset tracking.
- val: 25 episodes from the HF "train"-labeled pool, drawn from the remainder NOT selected into
  train (a later slice of the SAME shuffle) -- disjoint from train/train_eval by construction,
  not by a post-hoc set-difference check.
- test: 25 episodes from the HF "eval"-labeled pool -- a wholly separate, officially held-out
  pool the training episodes are never drawn from.

Output shape matches the existing oasis_scaling_*ep.json files exactly --
{"train": [...], "train_eval": [...], "val": [...], "test": [...]} with level_ids as bare episode
directory names (matching splits.json's own "name" field) -- so CraftiumOasisDataset's existing
load_split()/--split-path mechanism works completely unchanged, once CraftiumOasisDataset's
DATASET_ROOT is pointed at the extracted archive's OpenWorldCreative-v0/ directory.

Example:
    python -m scripts.make_luanti_1000ep_split
"""
import json
import os
import random
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    splits_path: str = "data/luanti/motworld-s1nav-10k.splits.json"
    output_path: str = "outputs/oasis_luanti_1000ep.json"
    num_train: int = 1000
    num_val: int = 25
    num_test: int = 25
    train_eval_cap: int = 25
    seed: int = 0


def main(args: Args):
    with open(args.splits_path) as f:
        splits = json.load(f)

    train_pool = [e["name"] for e in splits["episodes"] if e["split"] == "train"]
    eval_pool = [e["name"] for e in splits["episodes"] if e["split"] == "eval"]

    rng = random.Random(args.seed)
    shuffled_train = train_pool.copy()
    rng.shuffle(shuffled_train)
    shuffled_eval = eval_pool.copy()
    rng.shuffle(shuffled_eval)

    assert args.num_train + args.num_val <= len(shuffled_train), (
        f"train ({args.num_train}) + val ({args.num_val}) exceeds the HF 'train'-labeled pool "
        f"({len(shuffled_train)} episodes)"
    )
    assert args.num_test <= len(shuffled_eval), (
        f"test ({args.num_test}) exceeds the HF 'eval'-labeled pool ({len(shuffled_eval)} episodes)"
    )

    # Two disjoint slices of the SAME shuffle -- train and val can never overlap by construction,
    # not by a post-hoc check of an independently-drawn sample.
    train_ids = sorted(shuffled_train[: args.num_train])
    train_eval_ids = sorted(shuffled_train[: min(args.num_train, args.train_eval_cap)])
    val_ids = sorted(shuffled_train[args.num_train: args.num_train + args.num_val])
    test_ids = sorted(shuffled_eval[: args.num_test])

    # Belt-and-suspenders: train/test and val/test are drawn from disjoint HF-labeled pools
    # (train-labeled vs eval-labeled) so overlap there should be structurally impossible -- still
    # checked explicitly rather than assumed, same spirit as make_oasis_scaling_subsets.py's own
    # parent-split disjointness assert.
    assert set(train_ids).isdisjoint(val_ids), "train/val overlap -- bug in slicing"
    assert set(train_ids).isdisjoint(test_ids), "train/test overlap -- should be impossible (different HF pools)"
    assert set(val_ids).isdisjoint(test_ids), "val/test overlap -- should be impossible (different HF pools)"

    out = {
        "seed": args.seed,
        "num_train": args.num_train,
        "source": "percool/luanti motworld-s1nav-10k",
        "train": train_ids,
        "train_eval": train_eval_ids,
        "val": val_ids,
        "test": test_ids,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"train={len(train_ids)} train_eval={len(train_eval_ids)} val={len(val_ids)} "
          f"test={len(test_ids)} -> {args.output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
