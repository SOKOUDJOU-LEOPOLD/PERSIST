"""Deterministic train/test(/val) split of persist-eval-sample's 256 episodes, for fine-tuning
Oasis and WorldMem on real Craftium data (see baselines/finetune_oasis.py and the corresponding
WorldMem phase). Written once here and reused by both baselines' fine-tuning + evaluation
scripts.

Fixed seed, shuffle-then-slice: avoids any accidental ordering bias from the dataset's own
level_001..level_256 numbering (e.g. if levels were generated in some non-random world-difficulty
or chronological order) while still being exactly reproducible.

num_val=0 (the default) reproduces the original 236 train / 20 test split exactly as used
elsewhere in this project / referenced in the paper write-up -- do not change that default, or
regenerate outputs/finetune_split.json, without good reason. For a genuinely 3-way split (e.g.
the Oasis data-scaling study's 206/25/25), pass --num-val and write to a DIFFERENT --output-path
-- see scripts/make_oasis_scaling_subsets.py, which consumes this file's "train" list.

Example:
    python -m scripts.make_finetune_split --output-path outputs/finetune_split.json
    python -m scripts.make_finetune_split --num-val 25 --num-test 25 \
        --output-path outputs/finetune_split_oasis_scaling.json
"""
import json
import random
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    num_episodes: int = 256
    num_test: int = 20
    num_val: int = 0
    """0 (default) reproduces the original 2-way split with no val partition. Set >0 for a 3-way
    train/val/test split (adds a "val" key to the output json)."""
    seed: int = 0
    output_path: str = "outputs/finetune_split.json"


def main(args: Args):
    level_ids = [f"level_{i+1:03d}" for i in range(args.num_episodes)]
    rng = random.Random(args.seed)
    shuffled = level_ids.copy()
    rng.shuffle(shuffled)

    test_ids = sorted(shuffled[: args.num_test])
    val_ids = sorted(shuffled[args.num_test : args.num_test + args.num_val])
    train_ids = sorted(shuffled[args.num_test + args.num_val :])
    assert len(train_ids) + len(val_ids) + len(test_ids) == args.num_episodes
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(val_ids).isdisjoint(test_ids)

    split = {
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "train": train_ids,
        "test": test_ids,
    }
    if args.num_val > 0:
        split["val"] = val_ids
    with open(args.output_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Train: {len(train_ids)} episodes, Val: {len(val_ids)} episodes, Test: {len(test_ids)} episodes")
    if val_ids:
        print(f"Val set: {val_ids}")
    print(f"Test set: {test_ids}")
    print(f"Saved split to {args.output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
