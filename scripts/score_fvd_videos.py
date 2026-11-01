"""Score FVD between already-saved PERSIST rollout videos and their ground-truth episodes.

Complements `scripts/eval_fvd.py`: that script rolls out AND scores in one in-memory pass,
without ever writing an mp4 to disk. This script instead reads back rollout videos already
saved by `scripts/run_inference.py` (one `<variant>_<sha256>.mp4` per episode) together with
the dataset's own ground-truth `rgb.mp4` for the same episode. Separating generation (slow,
GPU-bound, can crash mid-run) from scoring (fast, cheap, rerunnable) means a crashed/resumed
rollout job never has to be regenerated just to get a score, and the videos themselves are kept
as durable, inspectable artifacts.

Frame alignment: `MinetestLatentCameraActionEval` (used by `run_inference.py`) always samples
ground truth starting at frame 0 of the episode (`sample_from_start_of_episode=True`,
`pixel_offset_ts=0`), so generated frame i aligns with ground-truth frame i directly -- no
offset to account for here. Both videos are decoded frame-by-frame with PyAV (bounds memory
for 600-frame clips) and truncated to each requested length before FVD scoring, matching
`utils/fvd_metric.py`'s `clips_from_video` (non-overlapping 16-frame clips, leftovers dropped)
and `scripts/eval_fvd.py`'s `to_uint8_range` convention ([0, 255] float into the I3D extractor).

Example:
    python -m scripts.score_fvd_videos \
        --dataset-path /data/lsokoudj/eval/persist_eval \
        --rollout-dir /data/lsokoudj/eval/rollouts/persist_eval/w_v0 \
        --frame-lengths 200 400 600 \
        --output-dir /data/lsokoudj/eval/fvd/persist_eval/w_v0

    # Sanity check the scorer itself before trusting it on real rollouts:
    python -m scripts.score_fvd_videos --dataset-path /data/lsokoudj/eval/persist_eval --self-test
"""
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av
import numpy as np
import pandas as pd
import torch
from loguru import logger

from utils.fvd_metric import (
    I3D_FEATURE_DIM,
    clips_from_video,
    compute_frechet_distance,
    extract_i3d_features,
    load_i3d_model,
)

ROLLOUT_NAME_RE = re.compile(r"^([A-Za-z0-9]+)_([0-9a-f]{64})\.mp4$")


@dataclass
class Args:
    dataset_path: str
    """Root of the dataset the rollouts were generated from (has metadata.csv + <local_path>/rgb.mp4)."""
    rollout_dir: Optional[str] = None
    """Directory of `<variant>_<sha256>.mp4` rollout videos saved by scripts/run_inference.py.
    Required unless --self-test is set."""
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])

    output_dir: Optional[str] = None
    """Where to write fvd.json / per_episode.csv. Required unless --self-test is set."""
    set_name: Optional[str] = None
    """Label for fvd.json's "set" field. Defaults to rollout_dir's grandparent directory name."""
    mode_name: Optional[str] = None
    """Label for fvd.json's "mode" field. Defaults to rollout_dir's own directory name."""

    i3d_checkpoint: Optional[str] = None
    i3d_batch_size: int = 8
    device: str = "cuda:0"

    self_test: bool = False
    """Sanity-check the scorer instead of scoring real rollouts: compute FVD(GT, GT) (expect ~0)
    and FVD(GT, GT shifted 8px) (expect clearly > 0) over real episodes. Run this once before
    trusting the real scoring below.

    Needs enough clips that the 400-dim I3D feature covariance isn't underdetermined (N must be
    well above 400, the feature dimensionality, or compute_frechet_distance's covariance
    estimate is near-singular and the resulting FVD is dominated by sampling noise rather than
    real distributional difference -- this is a generic small-sample FID/FVD pitfall, not
    specific to this implementation). The defaults below give ~2560 clips total (1280 per half),
    comfortably above that floor."""
    self_test_episodes: int = -1
    """Number of validation episodes to self-test with. -1 (default) uses all of them."""
    self_test_frames: int = 160


def read_video_frames(path: str, max_frames: int) -> Optional[torch.Tensor]:
    """Decode up to `max_frames` RGB frames from `path`, frame-by-frame (PyAV), as (T, C, H, W)
    float32 in [0, 255]. Returns None if the file can't be opened or has zero frames."""
    try:
        container = av.open(path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not open {path}: {e}")
        return None
    frames = []
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    finally:
        container.close()
    if not frames:
        return None
    video = np.stack(frames, axis=0)  # (T, H, W, C) uint8
    return torch.from_numpy(video).permute(0, 3, 1, 2).float()  # (T, C, H, W)


def gt_path_for(metadata: pd.DataFrame, dataset_path: str, sha256: str) -> Optional[str]:
    if sha256 not in metadata.index:
        return None
    local_path = metadata.loc[sha256, "local_path"]
    return os.path.join(dataset_path, local_path, "rgb.mp4")


def discover_rollouts(rollout_dir: str) -> Dict[str, str]:
    """Return {sha256: full_path} for every `<variant>_<sha256>.mp4` in rollout_dir."""
    out = {}
    for name in sorted(os.listdir(rollout_dir)):
        m = ROLLOUT_NAME_RE.match(name)
        if m:
            out[m.group(2)] = os.path.join(rollout_dir, name)
    return out


def run_self_test(args: Args):
    metadata = pd.read_csv(os.path.join(args.dataset_path, "metadata.csv")).set_index("sha256")
    val = metadata[metadata["validation_set"] == True]  # noqa: E712
    shas = val.index.tolist()
    # Shuffle (fixed seed) before any slicing/splitting: metadata.csv's row order often groups
    # episodes by scene/map, so splitting by file order can hand the two "halves" systematically
    # different content -- a real domain gap unrelated to what this test is trying to measure,
    # which is whether the scorer can tell "same distribution" from "shifted by 8px".
    random.Random(0).shuffle(shas)
    if args.self_test_episodes >= 0:
        shas = shas[: args.self_test_episodes]
    if not shas:
        raise ValueError(f"No validation episodes found in {args.dataset_path}/metadata.csv")

    i3d = load_i3d_model(args.i3d_checkpoint, device=args.device)
    gt_feats, noise_feats = [], []
    rng = np.random.RandomState(0)
    for sha in shas:
        path = gt_path_for(metadata, args.dataset_path, sha)
        video = read_video_frames(path, args.self_test_frames)
        if video is None or video.shape[0] < args.self_test_frames:
            logger.warning(f"Skipping {sha[:12]} (couldn't decode {args.self_test_frames} frames)")
            continue
        # A deliberately extreme, unambiguous negative control: i.i.d. random noise frames, same
        # shape as the real video. An earlier version of this test used an 8px pixel shift instead
        # (a much subtler perturbation) and it failed -- not because of a scoring bug, but because
        # at this sample size (~256 episodes, 400-dim I3D features) Frechet-distance estimators
        # carry a real, substantial finite-sample upward bias (a documented FID/FVD property, see
        # Chong & Forsyth 2020 "Effectively Unbiased FID/FVD Evaluation"), which swamps a shift
        # that small. Random noise is different enough from real video to be unmistakable
        # regardless of that bias, so it actually tests whether the pipeline is wired correctly.
        noise = torch.from_numpy(rng.randint(0, 256, size=tuple(video.shape), dtype=np.uint8)).float()
        gt_feats.append(extract_i3d_features(clips_from_video(video), i3d, args.device, args.i3d_batch_size))
        noise_feats.append(
            extract_i3d_features(clips_from_video(noise), i3d, args.device, args.i3d_batch_size)
        )
    gt_feats = np.concatenate(gt_feats, axis=0)
    noise_feats = np.concatenate(noise_feats, axis=0)
    # Split the GT set itself in half for a genuine two-sample FVD(GT, GT) rather than comparing
    # a distribution to itself exactly (which is trivially and uninformatively 0 by construction).
    # This number won't be ~0 at this sample size (see the bias note above) -- it's the noise
    # floor the real eval's FVD numbers should be read against, not a pass/fail threshold itself.
    half = len(gt_feats) // 2
    fvd_gt_gt = compute_frechet_distance(gt_feats[:half], gt_feats[half:])
    fvd_gt_noise = compute_frechet_distance(gt_feats, noise_feats)
    logger.info(f"Self-test over {len(shas)} episodes, {args.self_test_frames} frames each:")
    logger.info(f"  FVD(GT half A, GT half B)  = {fvd_gt_gt:.2f}  (finite-sample noise floor, not ~0)")
    logger.info(f"  FVD(GT, random noise)      = {fvd_gt_noise:.2f}  (expect orders of magnitude larger)")
    if not (fvd_gt_noise > 10 * fvd_gt_gt):
        logger.error("Self-test FAILED: FVD(GT, noise) is not decisively larger than the GT-vs-GT noise floor.")
        sys.exit(1)
    logger.info("Self-test passed.")


def main(args: Args):
    if args.self_test:
        run_self_test(args)
        return

    if not args.rollout_dir or not args.output_dir:
        raise ValueError("--rollout-dir and --output-dir are required unless --self-test is set.")

    metadata = pd.read_csv(os.path.join(args.dataset_path, "metadata.csv")).set_index("sha256")
    rollouts = discover_rollouts(args.rollout_dir)
    if not rollouts:
        raise ValueError(f"No `<variant>_<sha256>.mp4` files found in {args.rollout_dir}")
    logger.info(f"Found {len(rollouts)} rollout video(s) in {args.rollout_dir}")

    max_frames = max(args.frame_lengths)
    i3d = load_i3d_model(args.i3d_checkpoint, device=args.device)

    gen_feats: Dict[int, List[np.ndarray]] = {L: [] for L in args.frame_lengths}
    real_feats: Dict[int, List[np.ndarray]] = {L: [] for L in args.frame_lengths}
    rows = []

    for sha, gen_path in rollouts.items():
        gt_path = gt_path_for(metadata, args.dataset_path, sha)
        local_path = metadata.loc[sha, "local_path"] if sha in metadata.index else None
        if gt_path is None or not os.path.exists(gt_path):
            logger.warning(f"No ground-truth video for {sha[:12]} (local_path={local_path}); skipping")
            rows.append({"sha256": sha, "episode": local_path, "frames_gen": 0, "frames_gt": 0})
            continue

        gen = read_video_frames(gen_path, max_frames)
        real = read_video_frames(gt_path, max_frames)
        n_gen = gen.shape[0] if gen is not None else 0
        n_real = real.shape[0] if real is not None else 0
        rows.append({"sha256": sha, "episode": local_path, "frames_gen": n_gen, "frames_gt": n_real})

        if gen is None or real is None:
            logger.warning(f"Skipping {sha[:12]}: decode failed (gen={n_gen}, gt={n_real} frames)")
            continue
        if n_gen < max_frames or n_real < max_frames:
            logger.warning(
                f"Skipping {sha[:12]}: need >= {max_frames} frames in both, "
                f"got gen={n_gen}, gt={n_real}"
            )
            continue

        for length in args.frame_lengths:
            gen_clips = clips_from_video(gen[:length])
            real_clips = clips_from_video(real[:length])
            gen_feats[length].append(extract_i3d_features(gen_clips, i3d, args.device, args.i3d_batch_size))
            real_feats[length].append(extract_i3d_features(real_clips, i3d, args.device, args.i3d_batch_size))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "per_episode.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sha256", "episode", "frames_gen", "frames_gt"])
        writer.writeheader()
        writer.writerows(rows)

    fvd_by_length, n_clips_by_length = {}, {}
    for length in sorted(args.frame_lengths):
        g = (
            np.concatenate(gen_feats[length], axis=0)
            if gen_feats[length]
            else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        r = (
            np.concatenate(real_feats[length], axis=0)
            if real_feats[length]
            else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        if len(g) == 0 or len(r) == 0:
            logger.error(f"No usable episodes at length {length}; skipping FVD for this length.")
            continue
        fvd = compute_frechet_distance(r, g)
        fvd_by_length[str(length)] = fvd
        n_clips_by_length[str(length)] = int(g.shape[0])
        logger.info(f"FVD @ {length} frames: {fvd:.2f} ({g.shape[0]} clips from {len(rollouts)} episodes)")

    set_name = args.set_name or Path(args.rollout_dir).resolve().parent.name
    mode_name = args.mode_name or Path(args.rollout_dir).resolve().name
    result = {
        "set": set_name,
        "mode": mode_name,
        "n_episodes": len(rollouts),
        "fvd": fvd_by_length,
        "n_clips": n_clips_by_length,
    }
    out_json = os.path.join(args.output_dir, "fvd.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved {out_json}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
