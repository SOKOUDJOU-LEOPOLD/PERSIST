"""Deterministic train/test split of persist-eval-sample's 256 episodes, for fine-tuning Oasis
and WorldMem on real Craftium data (see baselines/finetune_oasis.py and the corresponding WorldMem
phase). Written once here and reused by both baselines' fine-tuning + evaluation scripts, so the
same 236/20 split is used throughout -- this is the actual data referenced in the paper write-up.

Fixed seed, shuffle-then-slice: avoids any accidental ordering bias from the dataset's own
level_001..level_256 numbering (e.g. if levels were generated in some non-random world-difficulty
or chronological order) while still being exactly reproducible.

Example:
    python -m scripts.make_finetune_split --output-path outputs/finetune_split.json
"""
import json
import random
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    num_episodes: int = 256
    num_test: int = 20
    seed: int = 0
    output_path: str = "outputs/finetune_split.json"


def main(args: Args):
    level_ids = [f"level_{i+1:03d}" for i in range(args.num_episodes)]
    rng = random.Random(args.seed)
    shuffled = level_ids.copy()
    rng.shuffle(shuffled)

    test_ids = sorted(shuffled[: args.num_test])
    train_ids = sorted(shuffled[args.num_test :])
    assert len(train_ids) + len(test_ids) == args.num_episodes
    assert set(train_ids).isdisjoint(test_ids)

    split = {
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "train": train_ids,
        "test": test_ids,
    }
    with open(args.output_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Train: {len(train_ids)} episodes, Test: {len(test_ids)} episodes")
    print(f"Test set: {test_ids}")
    print(f"Saved split to {args.output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
