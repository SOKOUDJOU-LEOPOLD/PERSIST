"""FVD for the fine-tuned Oasis checkpoint (real Craftium seed frame + real Craftium actions per
episode), on either the 236-episode train split or the 20-episode held-out test split from
outputs/finetune_split.json.

Protocol matches baselines/eval_oasis_zeroshot.py exactly (the same script used for the original
zero-shot Table 1 row, outputs/eval_fvd/oasis.json) so the two are as directly comparable as
possible given the different action-driving methodology (real actions here vs. Oasis's own
zero-shot action sampling there): ONE rollout per episode at max_frames=max(frame_lengths)=600,
then FVD computed three times by truncating that same rollout + the same real GT video to
[200, 400, 600] frames each -- not three separate generations. See eval_oasis_zeroshot.py:205-268
for the reference pattern this mirrors.

Example:
    cd baselines
    python eval_oasis_finetuned_fvd.py --split test \
        --ckpt-path ../outputs/finetune_oasis_v4/step_000500_verified.pt.bak
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from finetune_oasis import CRAFTIUM_ACTION_DIM, load_finetuned_dit  # noqa: E402
from finetune_oasis_data import DATASET_ROOT, load_split  # noqa: E402
from finetune_oasis_rollout import generate_rollout, shift_actions  # noqa: E402

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


@dataclass
class Args:
    split_path: str = "../outputs/finetune_split.json"
    split: Literal["train", "test"] = "test"
    ckpt_path: str = "../outputs/finetune_oasis_v4/step_000500_verified.pt.bak"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    dataset_root: str = DATASET_ROOT
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    ddim_steps: int = 10
    max_episodes: int = 0
    """If >0, cap the number of episodes processed -- 0 means all episodes in the split."""
    i3d_batch_size: int = 8
    output_json: str = "../outputs/eval_fvd/oasis_finetuned.json"
    device: str = "cuda:0"
    seed: int = 0


load_finetuned_model = load_finetuned_dit  # auto-detects gated vs. plain external_cond


def read_full_video(path: str) -> np.ndarray:
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    container.close()
    return np.stack(frames)  # (T, H, W, 3) uint8


def main(args: Args):
    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    model, step = load_finetuned_model(args.ckpt_path, device)
    print(f"Loaded checkpoint at step={step} from {args.ckpt_path}")

    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    i3d = load_i3d_model(device=device)

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
        npz = np.load(os.path.join(level_dir, "data.npz"))
        raw_action = torch.from_numpy(npz["action"][:max_frames].astype(np.float32))
        actions = shift_actions(raw_action)

        seed_frame = torch.from_numpy(real_video[0]).float().permute(2, 0, 1) / 255.0
        gen = generate_rollout(
            model, vae, seed_frame, actions, max_frames, device,
            ddim_steps=args.ddim_steps, show_progress=False,
        )  # (T, H, W, 3) uint8, CPU

        real = torch.from_numpy(real_video).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        gen = gen.permute(0, 3, 1, 2).float()

        n = min(gen.shape[0], real.shape[0])
        if n < max_frames:
            print(f"WARNING: {level_id}: only {n} frames available (wanted {max_frames}); truncating.")
        gen, real = gen[:n], real[:n]

        for length in args.frame_lengths:
            if gen.shape[0] < length:
                continue
            gen_clips = clips_from_video(gen[:length])
            real_clips = clips_from_video(real[:length])
            local_gen_feats[length].append(
                extract_i3d_features(gen_clips, i3d, device=device, batch_size=args.i3d_batch_size)
            )
            local_real_feats[length].append(
                extract_i3d_features(real_clips, i3d, device=device, batch_size=args.i3d_batch_size)
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
        "method": "Oasis (fine-tuned, 23-dim Craftium actions, real-action-driven)",
        "checkpoint": args.ckpt_path,
        "checkpoint_step": step,
        "split": args.split,
        "num_episodes": len(level_ids),
        "frame_lengths": args.frame_lengths,
        "fvd": fvd_by_length,
        "device": torch.cuda.get_device_name(device),
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
