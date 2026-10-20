"""Merges eval_oasis_finetuned_fvd_shard.py's per-EPISODE raw I3D + FID features (one small .npz
per completed episode, written by however many dynamic work-queue workers actually ran -- any
mix of GPU types/counts, see that script's module docstring) into one pool, and computes FVD and
FID exactly once each from the combined set -- mathematically identical to running every episode
through eval_oasis_finetuned_fvd.py in a single process, since that script also only ever computes
each Frechet distance once, from features concatenated across every episode in the split. Frechet
distance is NOT the average of several per-episode/per-worker FVD/FID values -- it must be
computed from one pooled feature set, which is why this merge step (not a simple average) is the
correct way to combine parallelized work.

Example:
    cd baselines
    python merge_oasis_fvd_shards.py \\
        --episode-feats-glob '../outputs/finetune_oasis_fullscale/episode_feats/*.npz' \\
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

_fid_metric_spec = importlib.util.spec_from_file_location(
    "persist_fid_metric", os.path.join(PERSIST_ROOT, "utils", "fid_metric.py")
)
_fid_metric = importlib.util.module_from_spec(_fid_metric_spec)
_fid_metric_spec.loader.exec_module(_fid_metric)
FID_FEATURE_DIM = _fid_metric.FID_FEATURE_DIM


@dataclass
class Args:
    episode_feats_glob: str = "../outputs/episode_feats/*.npz"
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    checkpoint: str = ""
    checkpoint_step: int = 0
    split: str = "train"
    output_json: str = "../outputs/eval_fvd_merged.json"


def main(args: Args):
    episode_paths = sorted(glob.glob(args.episode_feats_glob))
    if not episode_paths:
        raise ValueError(f"No episode feature files matched {args.episode_feats_glob!r}")
    print(f"Merging {len(episode_paths)} episodes")

    all_level_ids = []
    fvd_by_length = {}
    fid_by_length = {}
    psnr_by_length = {}

    for length in args.frame_lengths:
        gen_parts, real_parts, psnr_parts = [], [], []
        gen_fid_parts, real_fid_parts = [], []
        for path in episode_paths:
            d = np.load(path, allow_pickle=True)
            gen_parts.append(d[f"gen_feats_{length}"])
            real_parts.append(d[f"real_feats_{length}"])
            gen_fid_parts.append(d[f"gen_fid_feats_{length}"])
            real_fid_parts.append(d[f"real_fid_feats_{length}"])
            psnr_parts.append(d[f"psnr_values_{length}"])
            if length == args.frame_lengths[0]:
                all_level_ids.append(str(d["level_id"]))

        gen_feats = np.concatenate(gen_parts, axis=0) if gen_parts else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        real_feats = np.concatenate(real_parts, axis=0) if real_parts else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        gen_fid_feats = np.concatenate(gen_fid_parts, axis=0) if gen_fid_parts else np.zeros((0, FID_FEATURE_DIM), dtype=np.float32)
        real_fid_feats = np.concatenate(real_fid_parts, axis=0) if real_fid_parts else np.zeros((0, FID_FEATURE_DIM), dtype=np.float32)
        psnr_values = np.concatenate(psnr_parts, axis=0) if psnr_parts else np.zeros((0,), dtype=np.float32)

        fvd = compute_frechet_distance(real_feats, gen_feats)
        fid = compute_frechet_distance(real_fid_feats, gen_fid_feats)
        fvd_by_length[length] = fvd
        fid_by_length[length] = fid
        psnr_by_length[length] = float(np.mean(psnr_values)) if len(psnr_values) else 0.0
        print(f"FVD @ {length} frames: {fvd:.4f} ({real_feats.shape[0]} real / {gen_feats.shape[0]} generated clips, "
              f"pooled from {len(episode_paths)} episodes)")
        print(f"FID @ {length} frames: {fid:.4f} ({real_fid_feats.shape[0]} real / {gen_fid_feats.shape[0]} generated frames, "
              f"pooled from {len(episode_paths)} episodes)")
        print(f"PSNR @ {length} frames: {psnr_by_length[length]:.2f}dB ({len(psnr_values)} frames pooled)")

    result = {
        "method": "Oasis (fine-tuned, real-action-driven)",
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "split": args.split,
        "num_episodes": len(all_level_ids),
        "frame_lengths": args.frame_lengths,
        "fvd": fvd_by_length,
        "fid": fid_by_length,
        "psnr_db": psnr_by_length,
        "num_episode_files_merged": len(episode_paths),
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved merged result to {args.output_json}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
