"""FVD for a fine-tuned (or untrained-surgery) WorldMem checkpoint, real Craftium seed frame +
real actions + real ground-truth poses per episode, on either the 236-episode train split or the
20-episode held-out test split from outputs/finetune_split.json.

Protocol matches eval_oasis_finetuned_fvd.py exactly (itself matching eval_oasis_zeroshot.py /
eval_worldmem_zeroshot.py's own pattern), for the same three-way comparison already run for
Oasis: ONE rollout per episode at max_frames=max(frame_lengths)=600, then FVD computed three
times by truncating that same rollout + the same real GT video to [200, 400, 600] frames each --
not three separate generations.

Generation goes through `validation_step()` via `build_worldmem_shell`/`load_single_episode_batch`
(baselines/finetune_worldmem_generate.py) -- including that function's `chunk_size=n_tokens`
override, the fix for the severe sampling-time corruption bug documented there. Without it these
numbers would be measuring that bug, not the model.

`load_single_episode_batch` appends `memory_condition_length` placeholder frames (all copies of
frame 0) after the real `num_frames` requested, matching CraftiumWorldMemDataset's own batch
layout; validation_step's frame-count therefore runs to `num_frames + memory_condition_length`,
generating `memory_condition_length` extra (bogus, unused) tail frames alongside the real ones --
this script drops those before scoring, and prepends the real (not generated) context frame(s)
so the scored sequence is exactly `max_frames` real-position frames long, matching real_video.

Example:
    cd baselines
    python eval_worldmem_finetuned_fvd.py --split test \
        --ckpt-path ../outputs/finetune_worldmem_v4/step_001000_verified.pt.bak
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Literal

import av
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorldMem"))

from finetune_worldmem_data import DATASET_ROOT, load_split  # noqa: E402
from finetune_worldmem_generate import build_worldmem_shell, load_single_episode_batch  # noqa: E402

# Same fix as eval_oasis_finetuned_fvd.py: open-oasis's own flat utils.py (never actually
# imported by this script, but WorldMem's own package tree also shadows `utils` in a way that
# breaks a plain `from utils.fvd_metric import ...`) -- load by file path instead.
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


@dataclass
class Args:
    split_path: str = "../outputs/finetune_split.json"
    split: Literal["train", "test"] = "test"
    ckpt_path: str = "../outputs/finetune_worldmem_v4/step_001000_verified.pt.bak"
    config_path: str = "configurations/huggingface.yaml"
    dataset_root: str = DATASET_ROOT
    memory_condition_length: int = 8
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    max_episodes: int = 0
    """If >0, cap the number of episodes processed -- 0 means all episodes in the split."""
    i3d_batch_size: int = 8
    output_json: str = "../outputs/eval_fvd/worldmem_finetuned.json"
    device: str = "cuda:0"
    seed: int = 0


def read_full_video(path: str) -> np.ndarray:
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    container.close()
    return np.stack(frames)  # (T, H, W, 3) uint8


def load_worldmem_checkpoint(args: Args):
    device = args.device
    worldmem = build_worldmem_shell(args.config_path, device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    worldmem.diffusion_model.load_state_dict(ckpt["model"])
    worldmem.diffusion_model.eval()

    from huggingface_hub import hf_hub_download

    cfg = OmegaConf.load(args.config_path)
    vae_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.vae_path.split("/")[:2]), filename="/".join(cfg.vae_path.split("/")[2:])
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    worldmem.vae.load_state_dict(vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt, strict=True)
    return worldmem, ckpt.get("step")


def generate_rollout(worldmem, args: Args, level_id: str, max_frames: int) -> np.ndarray:
    """One (max_frames, H, W, 3) uint8 rollout: real context frame(s) + validation_step's
    prediction for the rest, with the bogus memory-padding tail frames dropped."""
    video, actions, poses, timestamp = load_single_episode_batch(
        args.dataset_root, level_id, max_frames, args.memory_condition_length, args.seed
    )
    batch = (video.to(args.device), actions.to(args.device), poses.to(args.device), timestamp.to(args.device))

    worldmem.validation_step_outputs = []
    with torch.no_grad():
        worldmem.validation_step(batch, 0)
    xs_pred, _ = worldmem.validation_step_outputs[-1]  # (max_frames + memory_condition_length - n_ctx, 1, C, H, W)

    n_context_frames = worldmem.context_frames // worldmem.frame_stack
    real_pred = xs_pred[: max_frames - n_context_frames, 0]  # drop the bogus memory-padding tail
    pred_pixels = torch.clamp(real_pred, 0, 1) * 255.0
    pred_pixels = pred_pixels.permute(0, 2, 3, 1).byte().cpu().numpy()  # (max_frames - n_ctx, H, W, 3)

    context_pixels = (torch.clamp(video[0, :n_context_frames], 0, 1) * 255.0).permute(0, 2, 3, 1).byte().numpy()
    return np.concatenate([context_pixels, pred_pixels], axis=0)  # (max_frames, H, W, 3)


def main(args: Args):
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    worldmem, step = load_worldmem_checkpoint(args)
    print(f"Loaded checkpoint at step={step} from {args.ckpt_path}")

    i3d = load_i3d_model(device=args.device)

    level_ids = load_split(args.split_path, args.split)
    if args.max_episodes > 0:
        level_ids = level_ids[: args.max_episodes]
    max_frames = max(args.frame_lengths)
    print(f"Evaluating {len(level_ids)} episodes from split='{args.split}' at max_frames={max_frames}, "
          f"scored at frame_lengths={args.frame_lengths}")

    local_real_feats = {length: [] for length in args.frame_lengths}
    local_gen_feats = {length: [] for length in args.frame_lengths}

    t0 = time.time()
    for idx, level_id in enumerate(level_ids):
        level_dir = os.path.join(args.dataset_root, level_id)
        real_video = read_full_video(os.path.join(level_dir, "rgb.mp4"))[:max_frames]

        gen = generate_rollout(worldmem, args, level_id, max_frames)

        real = torch.from_numpy(real_video).permute(0, 3, 1, 2).float()
        gen_t = torch.from_numpy(gen).permute(0, 3, 1, 2).float()

        n = min(gen_t.shape[0], real.shape[0])
        if n < max_frames:
            print(f"WARNING: {level_id}: only {n} frames available (wanted {max_frames}); truncating.")
        gen_t, real = gen_t[:n], real[:n]

        for length in args.frame_lengths:
            if gen_t.shape[0] < length:
                continue
            gen_clips = clips_from_video(gen_t[:length])
            real_clips = clips_from_video(real[:length])
            local_gen_feats[length].append(
                extract_i3d_features(gen_clips, i3d, device=args.device, batch_size=args.i3d_batch_size)
            )
            local_real_feats[length].append(
                extract_i3d_features(real_clips, i3d, device=args.device, batch_size=args.i3d_batch_size)
            )

        elapsed = time.time() - t0
        print(f"[{idx+1}/{len(level_ids)}] {level_id} done ({elapsed:.1f}s elapsed, "
              f"{elapsed/(idx+1):.1f}s/episode avg)")

    fvd_by_length = {}
    for length in sorted(args.frame_lengths):
        gen_feats = (
            np.concatenate(local_gen_feats[length], axis=0)
            if local_gen_feats[length] else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        real_feats = (
            np.concatenate(local_real_feats[length], axis=0)
            if local_real_feats[length] else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        fvd = compute_frechet_distance(real_feats, gen_feats)
        fvd_by_length[length] = fvd
        print(f"FVD @ {length} frames: {fvd:.4f} ({real_feats.shape[0]} real / {gen_feats.shape[0]} generated clips)")

    elapsed = time.time() - t0
    result = {
        "method": "WorldMem (fine-tuned, 23-dim Craftium actions, real-action-driven)",
        "checkpoint": args.ckpt_path,
        "checkpoint_step": step,
        "split": args.split,
        "num_episodes": len(level_ids),
        "frame_lengths": args.frame_lengths,
        "fvd": fvd_by_length,
        "device": torch.cuda.get_device_name(args.device),
        "wall_clock_seconds": elapsed,
    }
    print(f"\n=== FVD result (for the paper's conditions log) ===")
    print(json.dumps(result, indent=2))

    out_path = args.output_json.replace(".json", f"_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
