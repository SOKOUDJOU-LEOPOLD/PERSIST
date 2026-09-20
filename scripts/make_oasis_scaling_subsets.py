"""Builds the 5 nested training-episode subsets for the Oasis data-scaling study (2, 10, 20, 100,
206 episodes), all drawn from -- and only from -- the "train" list of a single canonical 3-way
split (outputs/finetune_split_oasis_scaling.json, made by scripts/make_finetune_split.py with
--num-val 25 --num-test 25). The "val"/"test" lists in that file are never read for this and are
copied through unchanged into every output file, so every one of the 5 runs validates against the
exact same 25 episodes.

Nested by construction: one fixed shuffle of the 206 train episodes, then each subset is a prefix
of that shuffle -- the 2-episode run's data is a strict subset of the 10-episode run's, which is a
strict subset of the 20-episode run's, and so on. This isolates "what does adding more data do"
(each larger run's data is a strict superset of every smaller run's), rather than 5 independently
sampled sets that could differ for unrelated reasons.

Also writes a fixed "train_eval" list per subset -- one fixed episode per training episode, capped
at 25 -- used by finetune_oasis.py's train-eval tracking (the overfitting-onset signal: comparing
this against the val metrics on the same step axis). For subsets with <=25 episodes, train_eval is
just the whole training subset; for the 100- and 206-episode subsets it's a fixed 25-episode
prefix of that subset's own shuffle, so cost is comparable across all 5 runs regardless of
training-set size.

Output files: outputs/oasis_scaling_{2,10,20,100,206}ep.json, each shaped
{"train": [...], "train_eval": [...], "val": [...], "test": [...]} -- so
CraftiumOasisDataset's existing load_split()/--split-path mechanism works completely unchanged,
just pointed at a different file per run (split="train" for training, split="val" for validation,
split="train_eval" for the overfitting-onset check).

Example:
    python -m scripts.make_oasis_scaling_subsets \
        --input-path outputs/finetune_split_oasis_scaling.json \
        --output-dir outputs
"""
import json
import os
import random
from dataclasses import dataclass, field
from typing import List

import tyro


@dataclass
class Args:
    input_path: str = "outputs/finetune_split_oasis_scaling.json"
    output_dir: str = "outputs"
    subset_sizes: List[int] = field(default_factory=lambda: [2, 10, 20, 100, 206])
    train_eval_cap: int = 25
    seed: int = 0


def main(args: Args):
    with open(args.input_path) as f:
        parent = json.load(f)
    train_pool = parent["train"]
    val_ids = parent["val"]
    test_ids = parent["test"]
    assert set(train_pool).isdisjoint(val_ids) and set(train_pool).isdisjoint(test_ids), (
        "parent split's train/val/test overlap -- refusing to build subsets from a bad split file"
    )

    rng = random.Random(args.seed)
    shuffled = train_pool.copy()
    rng.shuffle(shuffled)

    for n in args.subset_sizes:
        assert n <= len(shuffled), f"subset size {n} exceeds the train pool ({len(shuffled)} episodes)"
        train_subset = sorted(shuffled[:n])
        train_eval = sorted(shuffled[: min(n, args.train_eval_cap)])

        out = {
            "seed": args.seed,
            "num_train": n,
            "train": train_subset,
            "train_eval": train_eval,
            "val": val_ids,
            "test": test_ids,
        }
        out_path = os.path.join(args.output_dir, f"oasis_scaling_{n}ep.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"{n:>3} episodes: train={len(train_subset)} train_eval={len(train_eval)} "
              f"val={len(val_ids)} test={len(test_ids)} -> {out_path}")

    # Verify nesting: each subset's train list is a subset of the next larger one's.
    sorted_sizes = sorted(args.subset_sizes)
    for smaller, larger in zip(sorted_sizes, sorted_sizes[1:]):
        smaller_ids = set(sorted(shuffled[:smaller]))
        larger_ids = set(sorted(shuffled[:larger]))
        assert smaller_ids <= larger_ids, f"{smaller}ep is not a subset of {larger}ep -- nesting broken"
    print(f"Nesting verified: {' subset of '.join(str(s) for s in sorted_sizes)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
