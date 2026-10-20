"""Stage 1 (GPU-bound, parallelize across the multi-GPU/multi-node work-queue): extracts per-frame
Inception (FID) features for both the real and already-generated video of each episode, saving
one small .npz per episode -- (num_frames, 2048) real + (num_frames, 2048) generated features.

Does NOT compute any Frechet distance here (that's Stage 2, merge_per_frame_fid.py, which is
CPU-bound -- scipy's sqrtm doesn't run on GPU, so there's nothing to parallelize across GPUs for
that step; more GPUs only speed up THIS extraction stage).

Reuses the exact same dynamic work-queue / atomic-claim pattern as
eval_oasis_finetuned_fvd_shard.py (see that file's module docstring) so this can be sharded
across A100/A40/RTX pools the same way. Much lighter per episode than the original generation
job -- no DiT/VAE/DDIM involved, just decoding two already-existing 600-frame videos and running
them through the Inception feature extractor.

Example -- launch as many times as you want, any GPU type (see submit_per_frame_fid.sh):
    cd baselines
    python extract_per_frame_fid_features.py \\
        --dataset-root <dataset dir> --all-episodes \\
        --generated-video-dir ../outputs/.../videos \\
        --num-frames 600 \\
        --output-dir ../outputs/.../per_frame_fid_feats \\
        --device cuda:0
"""
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_action_layout import list_all_levels, read_video  # noqa: E402
from finetune_oasis_data import load_split  # noqa: E402

import importlib.util  # noqa: E402

PERSIST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    num_frames: int = 600
    fid_batch_size: int = 64
    output_dir: str = "per_frame_fid_feats"
    """One small {level_id}.npz written per completed episode -- same claim/completion pattern
    as eval_oasis_finetuned_fvd_shard.py's --output-dir (a .claims/ subdir here too)."""
    device: str = "cuda:0"


def claim_next(all_level_ids, output_dir, claims_dir):
    for level_id in all_level_ids:
        if os.path.exists(os.path.join(output_dir, f"{level_id}.npz")):
            continue
        try:
            os.makedirs(os.path.join(claims_dir, level_id), exist_ok=False)
        except FileExistsError:
            continue
        return level_id
    return None


@torch.no_grad()
def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)
    claims_dir = os.path.join(args.output_dir, ".claims")
    os.makedirs(claims_dir, exist_ok=True)

    if args.all_episodes:
        all_level_ids = list_all_levels(args.dataset_root)
    else:
        all_level_ids = load_split(args.split_path, args.split)
    print(f"Pool size {len(all_level_ids)} episodes, {args.num_frames} frames each")

    inception = load_fid_inception(device=args.device)

    t0 = time.time()
    num_done = 0
    while True:
        level_id = claim_next(all_level_ids, args.output_dir, claims_dir)
        if level_id is None:
            print("No unclaimed episodes remain -- pool exhausted, exiting.")
            break

        real_video = read_video(os.path.join(args.dataset_root, level_id, "rgb.mp4"), args.num_frames)
        gen_video = read_video(os.path.join(args.generated_video_dir, f"{level_id}.mp4"), args.num_frames)
        n = min(real_video.shape[0], gen_video.shape[0], args.num_frames)
        if n < args.num_frames:
            print(f"WARNING: {level_id}: only {n} frames available (wanted {args.num_frames})")

        real_chw = torch.from_numpy(real_video[:n]).permute(0, 3, 1, 2)
        gen_chw = torch.from_numpy(gen_video[:n]).permute(0, 3, 1, 2)
        real_feats = extract_fid_features(real_chw, inception, device=args.device, batch_size=args.fid_batch_size)
        gen_feats = extract_fid_features(gen_chw, inception, device=args.device, batch_size=args.fid_batch_size)

        tmp_path = os.path.join(args.output_dir, f".{level_id}.tmp.npz")
        final_path = os.path.join(args.output_dir, f"{level_id}.npz")
        np.savez(tmp_path, real_feats=real_feats, gen_feats=gen_feats, num_frames=n)
        os.rename(tmp_path, final_path)

        num_done += 1
        elapsed = time.time() - t0
        print(f"[{num_done} done] {level_id} ({elapsed:.1f}s elapsed, {elapsed/num_done:.1f}s/episode avg)")

    print(f"Worker finished: {num_done} episodes completed")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
