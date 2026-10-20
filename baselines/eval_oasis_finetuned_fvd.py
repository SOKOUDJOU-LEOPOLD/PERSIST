"""FVD + FID + PSNR for the fine-tuned Oasis checkpoint (real seed frame + real actions per
episode), on either a train/test split file or (with --all-episodes) every level_* episode
directly under --dataset-root (for eval sets with no split json, e.g. persist-eval-sample).

FVD protocol matches baselines/eval_oasis_zeroshot.py exactly (the same script used for the
original zero-shot Table 1 row, outputs/eval_fvd/oasis.json) so the two are as directly
comparable as possible given the different action-driving methodology (real actions here vs.
Oasis's own zero-shot action sampling there): ONE rollout per episode at
max_frames=max(frame_lengths)=600, then FVD/FID computed three times by truncating that same
rollout + the same real GT video to [200, 400, 600] frames each -- not three separate
generations. See eval_oasis_zeroshot.py:205-268 for the reference pattern this mirrors.

FID is computed the same way as FVD (pooled features across all episodes, one Frechet-distance
call per frame length) but from per-FRAME Inception features (utils/fid_metric.py) rather than
per-16-frame-clip I3D features (utils/fvd_metric.py) -- FID has no temporal clipping, FVD does.

PSNR is computed from that SAME rollout (no extra generation cost) using the exact formula
scripts/eval_qualitative.py already uses elsewhere in this codebase (10*log10(255^2/MSE) on
uint8 pixels, per frame), pooled across all frames at each of the same [200, 400, 600] lengths.
Unlike FVD/FID (distributional, no frame correspondence needed), PSNR needs the generated video to
actually be trying to reproduce the same real trajectory frame-by-frame -- true here since real
actions drive generation.

Episodes are generated in BATCHES (generate_rollout_batch, --batch-size) rather than one at a
time -- see eval_oasis_finetuned_fvd_shard.py's module docstring for why (autoregressive
generation has no parallelism across time, so batching across independent episodes is the real
per-GPU throughput lever).

This script is for smoke tests / small subsets on one GPU. For the full persist-eval-sample run
(256 episodes), use eval_oasis_finetuned_fvd_shard.py sharded across parallel SLURM jobs instead
-- FVD/FID must be computed once over the FULLY pooled features across all episodes (not averaged
per-subset), so a single-process run over many episodes doesn't parallelize across GPUs the way
the shard/merge scripts do.

Example:
    cd baselines
    python eval_oasis_finetuned_fvd.py --split test \\
        --ckpt-path ../outputs/finetune_oasis_v4/step_000500_verified.pt.bak

    # persist-eval-sample smoke test, native action layout, with video saving:
    python eval_oasis_finetuned_fvd.py --all-episodes --max-episodes 2 --frame-lengths 20 40 \\
        --dataset-root <persist-eval-sample snapshot dir> \\
        --ckpt-path ../outputs/finetune_oasis_luanti_hudnav_1000ep/step_027500.pt \\
        --use-oasis-native-action-layout --video-output-dir /tmp/smoke/videos \\
        --output-json /tmp/smoke/smoke.json --device cuda:0
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from eval_action_layout import list_all_levels, load_eval_actions_and_video  # noqa: E402
from finetune_oasis import load_finetuned_dit  # noqa: E402
from finetune_oasis_data import DATASET_ROOT, load_split  # noqa: E402
from finetune_oasis_rollout import generate_rollout_batch  # noqa: E402

# Loaded by file path, not `from utils.fvd_metric import ...`: open-oasis's own flat utils.py
# (imported transitively via finetune_oasis_rollout -> finetune_oasis -> dit/vae) already
# occupies the `utils` name in sys.modules, so a package-style import of PERSIST's utils/
# package would fail with "'utils' is not a package" -- same fix as eval_oasis_zeroshot.py.
import importlib.util  # noqa: E402

PERSIST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fvd_metric_spec = importlib.util.spec_from_file_location(
    "persist_fvd_metric", os.path.join(PERSIST_ROOT, "utils", "fvd_metric.py")
)
_fvd_metric = importlib.util.module_from_spec(_fvd_metric_spec)
_fvd_metric_spec.loader.exec_module(_fvd_metric)
I3D_FEATURE_DIM = _fvd_metric.I3D_FEATURE_DIM
clips_from_video = _fvd_metric.clips_from_video
compute_frechet_distance = _fvd_metric.compute_frechet_distance
extract_i3d_features = _fvd_metric.extract_i3d_features
load_i3d_model = _fvd_metric.load_i3d_model

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
    split_path: str = "../outputs/finetune_split.json"
    split: str = "test"
    """Any key present in the split_path JSON -- e.g. "train"/"test" for the Luanti-specific
    split files, or "eval" for motworld-hudnav-20k's own held-out-episode labeling. Not
    restricted to a fixed Literal set since different split files use different label names."""
    all_episodes: bool = False
    """When True, ignore split_path/split and evaluate every level_* dir directly under
    dataset_root (for eval sets like persist-eval-sample that have no train/test split json)."""
    ckpt_path: str = "../outputs/finetune_oasis_v4/step_000500_verified.pt.bak"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    dataset_root: str = DATASET_ROOT
    use_oasis_native_action_layout: bool = False
    """Set True for a checkpoint trained with --use-oasis-native-action-layout (25-dim actions,
    e.g. the Luanti hudnav fine-tune) -- False (raw 23-dim) for older non-native checkpoints."""
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    ddim_steps: int = 10
    max_episodes: int = 0
    """If >0, cap the number of episodes processed -- 0 means all episodes in the split."""
    batch_size: int = 4
    """Episodes per generate_rollout_batch call -- see module docstring."""
    i3d_batch_size: int = 8
    fid_batch_size: int = 32
    video_output_dir: str = ""
    """When non-empty, saves each episode's generated max(frame_lengths)-frame rollout as
    {video_output_dir}/{level_id}.mp4 -- the SAME rollout used for scoring."""
    video_fps: int = 24
    output_json: str = "../outputs/eval_fvd/oasis_finetuned.json"
    device: str = "cuda:0"
    seed: int = 0


load_finetuned_model = load_finetuned_dit  # auto-detects gated vs. plain external_cond


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    """Verbatim copy of scripts/eval_qualitative.py's psnr() -- same formula, same 99.0 cap for
    a near-perfect match (avoids log(0)) -- reused rather than reinvented so this reports the
    same metric definition as the rest of the codebase."""
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def _chunked(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@torch.no_grad()
def main(args: Args):
    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    if args.video_output_dir:
        os.makedirs(args.video_output_dir, exist_ok=True)

    model, step = load_finetuned_model(args.ckpt_path, device)
    print(f"Loaded checkpoint at step={step} from {args.ckpt_path}")

    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    i3d = load_i3d_model(device=device)
    fid_inception = load_fid_inception(device=device)

    level_ids = list_all_levels(args.dataset_root) if args.all_episodes else load_split(args.split_path, args.split)
    if args.max_episodes > 0:
        level_ids = level_ids[: args.max_episodes]
    max_frames = max(args.frame_lengths)
    print(f"Evaluating {len(level_ids)} episodes at max_frames={max_frames}, "
          f"scored at frame_lengths={args.frame_lengths}, batch_size={args.batch_size}, "
          f"use_oasis_native_action_layout={args.use_oasis_native_action_layout}")

    local_real_feats = {length: [] for length in args.frame_lengths}
    local_gen_feats = {length: [] for length in args.frame_lengths}
    local_real_fid_feats = {length: [] for length in args.frame_lengths}
    local_gen_fid_feats = {length: [] for length in args.frame_lengths}
    psnr_values = {length: [] for length in args.frame_lengths}

    t0 = time.time()
    num_done = 0
    for batch_ids in _chunked(level_ids, args.batch_size):
        loaded = [
            load_eval_actions_and_video(args.dataset_root, lid, max_frames, args.use_oasis_native_action_layout)
            for lid in batch_ids
        ]
        avail_lens = [real.shape[0] for real, _, _ in loaded]
        action_dim = loaded[0][1].shape[-1]
        actions_batch = torch.zeros(len(loaded), max_frames, action_dim, dtype=torch.float32)
        seed_frames_batch = torch.stack([seed for _, _, seed in loaded], dim=0)
        for i, (_, actions, _) in enumerate(loaded):
            actions_batch[i, : actions.shape[0]] = actions

        gen_batch = generate_rollout_batch(
            model, vae, seed_frames_batch, actions_batch, max_frames, device,
            ddim_steps=args.ddim_steps, show_progress=True,
            desc=f"batch[{','.join(batch_ids)}]",
        )  # (B, max_frames, H, W, 3) uint8, CPU

        for i, level_id in enumerate(batch_ids):
            real_video = loaded[i][0]
            n = min(avail_lens[i], gen_batch.shape[1])
            if n < max_frames:
                print(f"WARNING: {level_id}: only {n} frames available (wanted {max_frames}); truncating.")
            gen_np = gen_batch[i, :n].numpy()
            real_np = real_video[:n]

            if args.video_output_dir:
                imageio.mimsave(os.path.join(args.video_output_dir, f"{level_id}.mp4"), gen_np, fps=args.video_fps)

            gen_chw = torch.from_numpy(gen_np).permute(0, 3, 1, 2).float()
            real_chw = torch.from_numpy(real_np).permute(0, 3, 1, 2).float()
            per_frame_psnr = [psnr(gen_np[t], real_np[t]) for t in range(n)]

            for length in args.frame_lengths:
                if n < length:
                    continue
                gen_clips = clips_from_video(gen_chw[:length])
                real_clips = clips_from_video(real_chw[:length])
                local_gen_feats[length].append(
                    extract_i3d_features(gen_clips, i3d, device=device, batch_size=args.i3d_batch_size)
                )
                local_real_feats[length].append(
                    extract_i3d_features(real_clips, i3d, device=device, batch_size=args.i3d_batch_size)
                )
                local_gen_fid_feats[length].append(
                    extract_fid_features(gen_chw[:length].byte(), fid_inception, device=device, batch_size=args.fid_batch_size)
                )
                local_real_fid_feats[length].append(
                    extract_fid_features(real_chw[:length].byte(), fid_inception, device=device, batch_size=args.fid_batch_size)
                )
                psnr_values[length].extend(per_frame_psnr[:length])

            num_done += 1
            elapsed = time.time() - t0
            print(f"[{num_done}/{len(level_ids)}] {level_id} done ({elapsed:.1f}s elapsed, "
                  f"{elapsed/num_done:.1f}s/episode avg)")

    def pooled_frechet(gen_dict, real_dict, dim, label):
        by_length = {}
        for length in sorted(args.frame_lengths):
            gen_feats = np.concatenate(gen_dict[length], axis=0) if gen_dict[length] else np.zeros((0, dim), dtype=np.float32)
            real_feats = np.concatenate(real_dict[length], axis=0) if real_dict[length] else np.zeros((0, dim), dtype=np.float32)
            d = compute_frechet_distance(real_feats, gen_feats)
            by_length[length] = d
            print(f"{label} @ {length} frames: {d:.4f} ({real_feats.shape[0]} real / {gen_feats.shape[0]} generated)")
        return by_length

    fvd_by_length = pooled_frechet(local_gen_feats, local_real_feats, I3D_FEATURE_DIM, "FVD")
    fid_by_length = pooled_frechet(local_gen_fid_feats, local_real_fid_feats, FID_FEATURE_DIM, "FID")

    psnr_by_length = {}
    for length in sorted(args.frame_lengths):
        mean_psnr = float(np.mean(psnr_values[length])) if psnr_values[length] else 0.0
        psnr_by_length[length] = mean_psnr
        print(f"PSNR @ {length} frames: {mean_psnr:.2f}dB ({len(psnr_values[length])} frames pooled)")

    elapsed = time.time() - t0
    result = {
        "method": "Oasis (fine-tuned, real-action-driven)",
        "checkpoint": args.ckpt_path,
        "checkpoint_step": step,
        "all_episodes": args.all_episodes,
        "split": args.split if not args.all_episodes else "all",
        "dataset_root": args.dataset_root,
        "num_episodes": len(level_ids),
        "frame_lengths": args.frame_lengths,
        "fvd": fvd_by_length,
        "fid": fid_by_length,
        "psnr_db": psnr_by_length,
        "device": torch.cuda.get_device_name(device),
        "wall_clock_seconds": elapsed,
    }
    print("\n=== FVD/FID result ===")
    print(json.dumps(result, indent=2))

    split_tag = args.split if not args.all_episodes else "all"
    out_path = args.output_json.replace(".json", f"_{split_tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
