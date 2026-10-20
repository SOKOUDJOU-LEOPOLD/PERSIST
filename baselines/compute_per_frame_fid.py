"""Per-frame FID: for each exact frame index t in [0, num_frames), pools ONLY frame t across
every episode (real vs. generated) and computes one Frechet distance from that pool -- a direct
"how degraded is frame t specifically" drift signal, as opposed to the cumulative FID@200/400/600
the main eval pipeline reports (which pools frames [0, t] together, conflating current quality
with everything before it).

Reuses the ALREADY-GENERATED 600-frame videos saved by eval_oasis_finetuned_fvd_shard.py's
--video-output-dir (no regeneration -- this is a much cheaper post-processing pass: decode
already-saved videos, extract per-frame Inception features, pool by exact frame index, done).
Real videos are decoded from the original dataset the same way eval_action_layout.py does.

Same compute_frechet_distance (utils/fvd_metric.py) and extract_fid_features/load_fid_inception
(utils/fid_metric.py) used everywhere else this session -- not a different formula.

Example:
    cd baselines
    python compute_per_frame_fid.py \\
        --dataset-root <persist-eval-sample dir> --all-episodes \\
        --generated-video-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/videos \\
        --num-frames 600 --output-csv ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/fid_per_frame.csv
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_action_layout import list_all_levels, read_video  # noqa: E402
from finetune_oasis_data import load_split  # noqa: E402

import importlib.util  # noqa: E402

PERSIST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fvd_metric_spec = importlib.util.spec_from_file_location(
    "persist_fvd_metric", os.path.join(PERSIST_ROOT, "utils", "fvd_metric.py")
)
_fvd_metric = importlib.util.module_from_spec(_fvd_metric_spec)
_fvd_metric_spec.loader.exec_module(_fvd_metric)
compute_frechet_distance = _fvd_metric.compute_frechet_distance

_fid_metric_spec = importlib.util.spec_from_file_location(
    "persist_fid_metric", os.path.join(PERSIST_ROOT, "utils", "fid_metric.py")
)
_fid_metric = importlib.util.module_from_spec(_fid_metric_spec)
_fid_metric_spec.loader.exec_module(_fid_metric)
FID_FEATURE_DIM = _fid_metric.FID_FEATURE_DIM
extract_fid_features = _fid_metric.extract_fid_features
load_fid_inception = _fid_metric.load_fid_inception


@dataclass
class Args:
    dataset_root: str = ""
    all_episodes: bool = False
    split_path: str = ""
    split: str = "eval"
    generated_video_dir: str = ""
    """Directory of already-generated {level_id}.mp4 files (from a prior
    eval_oasis_finetuned_fvd_shard.py --video-output-dir run)."""
    num_frames: int = 600
    max_episodes: int = 0
    """If >0, cap the number of episodes processed -- 0 means all episodes."""
    fid_batch_size: int = 64
    output_csv: str = "fid_per_frame.csv"
    device: str = "cuda:0"


@torch.no_grad()
def main(args: Args):
    if args.all_episodes:
        level_ids = list_all_levels(args.dataset_root)
    else:
        level_ids = load_split(args.split_path, args.split)
    if args.max_episodes > 0:
        level_ids = level_ids[: args.max_episodes]
    print(f"{len(level_ids)} episodes, {args.num_frames} frames each")

    inception = load_fid_inception(device=args.device)

    # (num_episodes, num_frames, FID_FEATURE_DIM) -- ~256*600*2048*4 bytes ~= 1.3GB per side for
    # persist256, well within host memory; feature dim is small even though we're holding every
    # frame of every episode, since Inception features (not raw pixels) are what's stored.
    real_feats = np.zeros((len(level_ids), args.num_frames, FID_FEATURE_DIM), dtype=np.float32)
    gen_feats = np.zeros((len(level_ids), args.num_frames, FID_FEATURE_DIM), dtype=np.float32)

    for i, level_id in enumerate(level_ids):
        real_video = read_video(os.path.join(args.dataset_root, level_id, "rgb.mp4"), args.num_frames)
        gen_video = read_video(os.path.join(args.generated_video_dir, f"{level_id}.mp4"), args.num_frames)
        n = min(real_video.shape[0], gen_video.shape[0], args.num_frames)
        if n < args.num_frames:
            print(f"WARNING: {level_id}: only {n} frames available (wanted {args.num_frames})")

        real_chw = torch.from_numpy(real_video[:n]).permute(0, 3, 1, 2)
        gen_chw = torch.from_numpy(gen_video[:n]).permute(0, 3, 1, 2)
        real_feats[i, :n] = extract_fid_features(real_chw, inception, device=args.device, batch_size=args.fid_batch_size)
        gen_feats[i, :n] = extract_fid_features(gen_chw, inception, device=args.device, batch_size=args.fid_batch_size)

        if (i + 1) % 10 == 0 or i == len(level_ids) - 1:
            print(f"[{i+1}/{len(level_ids)}] {level_id} features extracted")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        f.write("frame,fid\n")
        for t in range(args.num_frames):
            fid_t = compute_frechet_distance(real_feats[:, t, :], gen_feats[:, t, :])
            f.write(f"{t},{fid_t:.6f}\n")
            if t % 100 == 0:
                print(f"frame {t}: FID={fid_t:.4f}")

    print(f"Saved per-frame FID to {args.output_csv}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
