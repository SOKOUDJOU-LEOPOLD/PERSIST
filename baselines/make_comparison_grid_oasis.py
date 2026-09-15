"""Generates Oasis's half of a combined Oasis+WorldMem comparison grid: real footage vs.
untrained-surgery vs. fine-tuned, with per-frame PSNR against real footage, for a LIST of
episodes. Saves one .npz per episode (raw arrays, not a finished video) into --output-dir --
combine_comparison_grids.py stitches these together with make_comparison_grid_worldmem.py's
matching per-episode outputs into one multi-row grid video, since Oasis and WorldMem run in
separate Python environments and can't be loaded in the same process.

Each checkpoint is loaded ONCE and reused across all episodes (not reloaded per episode) --
loading dominates the per-call overhead relative to a single short rollout, so this matters when
generating many episodes.

Reuses generate_rollout (the exact same sampling function eval_oasis_finetuned_fvd.py uses for
the FVD/PSNR numbers), so this is the same model behavior those numbers describe, not a
different, cherry-picked generation path.

Example:
    cd baselines
    python make_comparison_grid_oasis.py --level-ids level_001,level_003,level_004 \
        --output-dir ../outputs/qualitative_grids
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List

import av
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from finetune_oasis import load_finetuned_dit  # noqa: E402
from finetune_oasis_data import DATASET_ROOT  # noqa: E402
from finetune_oasis_rollout import generate_rollout, shift_actions  # noqa: E402


@dataclass
class Args:
    level_ids: List[str] = field(default_factory=lambda: ["level_001"])
    dataset_root: str = DATASET_ROOT
    untrained_ckpt: str = "../outputs/finetune_oasis_v4/step_000000.pt"
    finetuned_ckpt: str = "../outputs/finetune_oasis_v4/step_000500_verified.pt.bak"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    num_frames: int = 150
    ddim_steps: int = 10
    fps: int = 20
    output_dir: str = "../outputs/qualitative_grids"
    device: str = "cuda:0"
    seed: int = 0


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def load_real_and_actions(dataset_root: str, level_id: str, num_frames: int):
    level_dir = os.path.join(dataset_root, level_id)
    container = av.open(os.path.join(level_dir, "rgb.mp4"))
    real_frames = []
    for i, frame in enumerate(container.decode(container.streams.video[0])):
        if i >= num_frames:
            break
        real_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    real = np.stack(real_frames)  # (T, H, W, 3) uint8

    npz = np.load(os.path.join(level_dir, "data.npz"))
    raw_action = torch.from_numpy(npz["action"][:num_frames].astype(np.float32))
    actions = shift_actions(raw_action)
    seed_frame = torch.from_numpy(real[0]).float().permute(2, 0, 1) / 255.0
    return real, actions, seed_frame


@torch.no_grad()
def main(args: Args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    # Pre-load real video + actions for every episode once (cheap, CPU-only).
    per_episode = {
        level_id: load_real_and_actions(args.dataset_root, level_id, args.num_frames)
        for level_id in args.level_ids
    }
    rollouts = {level_id: {} for level_id in args.level_ids}

    for name, ckpt_path in [("untrained", args.untrained_ckpt), ("finetuned", args.finetuned_ckpt)]:
        model, step = load_finetuned_dit(ckpt_path, device)
        print(f"[{name}] loaded checkpoint at step={step} from {ckpt_path}")
        for level_id in args.level_ids:
            torch.manual_seed(args.seed)
            _, actions, seed_frame = per_episode[level_id]
            gen = generate_rollout(
                model, vae, seed_frame, actions, args.num_frames, device,
                ddim_steps=args.ddim_steps, show_progress=True,
            ).numpy()  # (T, H, W, 3) uint8
            rollouts[level_id][name] = gen
            print(f"  [{name}] {level_id} done")
        del model
        torch.cuda.empty_cache()

    for level_id in args.level_ids:
        real, _, _ = per_episode[level_id]
        untrained, finetuned = rollouts[level_id]["untrained"], rollouts[level_id]["finetuned"]
        n = min(real.shape[0], untrained.shape[0], finetuned.shape[0])
        real_n, untrained_n, finetuned_n = real[:n], untrained[:n], finetuned[:n]
        psnr_untrained = np.array([psnr(untrained_n[t], real_n[t]) for t in range(n)], dtype=np.float32)
        psnr_finetuned = np.array([psnr(finetuned_n[t], real_n[t]) for t in range(n)], dtype=np.float32)

        out_path = os.path.join(args.output_dir, f"oasis_{level_id}.npz")
        np.savez(
            out_path,
            real=real_n, untrained=untrained_n, finetuned=finetuned_n,
            psnr_untrained=psnr_untrained, psnr_finetuned=psnr_finetuned,
            level_id=level_id, fps=args.fps,
        )
        print(f"Saved Oasis comparison data to {out_path} ({n} frames)")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
