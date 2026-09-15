"""Merges eval_oasis_finetuned_fvd_shard.py's per-shard raw I3D features into one pool and
computes FVD exactly once from the combined set -- mathematically identical to running every
episode through eval_oasis_finetuned_fvd.py in a single process, since that script also only
ever computes FVD once, from features concatenated across every episode in the split. Frechet
distance is NOT the average of several per-shard FVD values -- it must be computed from one
pooled feature set, which is why this merge step (not a simple average) is the correct way to
combine parallelized shards.

Example:
    cd baselines
    python merge_oasis_fvd_shards.py \\
        --shard-npz-glob '../outputs/finetune_oasis_fullscale/shard_feats/shard_*.npz' \\
        --checkpoint ../outputs/finetune_oasis_fullscale/step_010000.pt \\
        --checkpoint-step 10000 \\
        --split train \\
        --output-json ../outputs/finetune_oasis_fullscale/eval_fvd_train_full.json
"""
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

PERSIST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fvd_metric_spec = importlib.util.spec_from_file_location(
    "persist_fvd_metric", os.path.join(PERSIST_ROOT, "utils", "fvd_metric.py")
)
_fvd_metric = importlib.util.module_from_spec(_fvd_metric_spec)
_fvd_metric_spec.loader.exec_module(_fvd_metric)
I3D_FEATURE_DIM = _fvd_metric.I3D_FEATURE_DIM
compute_frechet_distance = _fvd_metric.compute_frechet_distance


@dataclass
class Args:
    shard_npz_glob: str = "../outputs/shard_feats/shard_*.npz"
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    checkpoint: str = ""
    checkpoint_step: int = 0
    split: str = "train"
    output_json: str = "../outputs/eval_fvd_merged.json"


def main(args: Args):
    shard_paths = sorted(glob.glob(args.shard_npz_glob))
    if not shard_paths:
        raise ValueError(f"No shard files matched {args.shard_npz_glob!r}")
    print(f"Merging {len(shard_paths)} shards: {shard_paths}")

    all_level_ids = []
    fvd_by_length = {}
    psnr_by_length = {}

    for length in args.frame_lengths:
        gen_parts, real_parts, psnr_parts = [], [], []
        for path in shard_paths:
            d = np.load(path, allow_pickle=True)
            gen_parts.append(d[f"gen_feats_{length}"])
            real_parts.append(d[f"real_feats_{length}"])
            psnr_parts.append(d[f"psnr_values_{length}"])
            if length == args.frame_lengths[0]:
                all_level_ids.extend(d["level_ids"].tolist())

        gen_feats = np.concatenate(gen_parts, axis=0) if gen_parts else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        real_feats = np.concatenate(real_parts, axis=0) if real_parts else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        psnr_values = np.concatenate(psnr_parts, axis=0) if psnr_parts else np.zeros((0,), dtype=np.float32)

        fvd = compute_frechet_distance(real_feats, gen_feats)
        fvd_by_length[length] = fvd
        psnr_by_length[length] = float(np.mean(psnr_values)) if len(psnr_values) else 0.0
        print(f"FVD @ {length} frames: {fvd:.4f} ({real_feats.shape[0]} real / {gen_feats.shape[0]} generated clips, "
              f"pooled from {len(shard_paths)} shards)")
        print(f"PSNR @ {length} frames: {psnr_by_length[length]:.2f}dB ({len(psnr_values)} frames pooled)")

    result = {
        "method": "Oasis (fine-tuned, 23-dim Craftium actions, real-action-driven)",
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "split": args.split,
        "num_episodes": len(all_level_ids),
        "frame_lengths": args.frame_lengths,
        "fvd": fvd_by_length,
        "psnr_db": psnr_by_length,
        "num_shards_merged": len(shard_paths),
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved merged result to {args.output_json}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
