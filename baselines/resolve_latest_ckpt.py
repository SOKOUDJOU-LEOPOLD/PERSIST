"""Resolves the latest step_NNNNNN.pt checkpoint in a training output dir. Run ONCE at submission
time (not once per shard) and pass the resolved path/step to every shard explicitly -- training
may still be ongoing, and SLURM array tasks can start hours apart (queue delays), so resolving
independently per-task risks two shards scoring two different checkpoints, silently corrupting
the pooled-feature merge (FVD/FID require every shard's features to come from the same model).

Example:
    cd baselines
    read CKPT_PATH CKPT_STEP < <(.venv/bin/python resolve_latest_ckpt.py --ckpt-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep)
"""
import glob
import os
import re
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Args:
    ckpt_dir: str = "../outputs/finetune_oasis_luanti_hudnav_1000ep"


def resolve_latest_checkpoint(ckpt_dir: str) -> Tuple[str, int]:
    paths = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No step_*.pt checkpoints found in {ckpt_dir}")

    def step_of(p: str) -> int:
        m = re.search(r"step_(\d+)\.pt$", p)
        if not m:
            raise ValueError(f"Checkpoint filename doesn't match step_NNNNNN.pt: {p}")
        return int(m.group(1))

    best = max(paths, key=step_of)
    return os.path.abspath(best), step_of(best)


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    path, step = resolve_latest_checkpoint(args.ckpt_dir)
    print(f"{path} {step}")
