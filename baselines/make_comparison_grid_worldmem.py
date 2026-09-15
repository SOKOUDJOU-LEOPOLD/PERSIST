"""Generates WorldMem's half of a combined Oasis+WorldMem comparison grid: real footage vs.
untrained-surgery vs. fine-tuned, with per-frame PSNR against real footage, for a LIST of
episodes. Saves one .npz per episode (raw arrays, not a finished video) into --output-dir --
combine_comparison_grids.py stitches these together with make_comparison_grid_oasis.py's
matching per-episode outputs into one multi-row grid video, since Oasis and WorldMem run in
separate Python environments and can't be loaded in the same process.

Each checkpoint is loaded ONCE and reused across all episodes (not reloaded per episode).

Reuses generate_rollout (the exact same validation_step-based sampling
eval_worldmem_finetuned_fvd.py uses for the FVD/PSNR numbers, chunk_size fix included via
build_worldmem_shell), so this is the same model behavior those numbers describe.

Example:
    cd baselines/WorldMem
    .venv/bin/python ../make_comparison_grid_worldmem.py --level-ids level_001,level_003 \
        --output-dir ../../outputs/qualitative_grids
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorldMem"))

from finetune_worldmem_data import DATASET_ROOT  # noqa: E402
from finetune_worldmem_generate import build_worldmem_shell, load_single_episode_batch  # noqa: E402


@dataclass
class Args:
    level_ids: List[str] = field(default_factory=lambda: ["level_001"])
    dataset_root: str = DATASET_ROOT
    untrained_ckpt: str = "../../outputs/finetune_worldmem_v4/step_000000.pt"
    finetuned_ckpt: str = "../../outputs/finetune_worldmem_v4/step_001000_verified.pt.bak"
    config_path: str = "configurations/huggingface.yaml"
    memory_condition_length: int = 8
    num_frames: int = 150
    fps: int = 20
    output_dir: str = "../../outputs/qualitative_grids"
    device: str = "cuda:0"
    seed: int = 0


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def load_checkpoint(config_path: str, ckpt_path: str, device: str):
    worldmem = build_worldmem_shell(config_path, device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    worldmem.diffusion_model.load_state_dict(ckpt["model"])
    worldmem.diffusion_model.eval()

    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download

    cfg = OmegaConf.load(config_path)
    vae_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.vae_path.split("/")[:2]), filename="/".join(cfg.vae_path.split("/")[2:])
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    worldmem.vae.load_state_dict(vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt, strict=True)
    return worldmem, ckpt.get("step")


@torch.no_grad()
def generate(worldmem, args: Args, level_id: str) -> np.ndarray:
    video, actions, poses, timestamp = load_single_episode_batch(
        args.dataset_root, level_id, args.num_frames, args.memory_condition_length, args.seed
    )
    batch = (video.to(args.device), actions.to(args.device), poses.to(args.device), timestamp.to(args.device))
    worldmem.validation_step_outputs = []
    worldmem.validation_step(batch, 0)
    xs_pred, _ = worldmem.validation_step_outputs[-1]

    n_context_frames = worldmem.context_frames // worldmem.frame_stack
    real_pred = xs_pred[: args.num_frames - n_context_frames, 0]
    pred_pixels = torch.clamp(real_pred, 0, 1) * 255.0
    pred_pixels = pred_pixels.permute(0, 2, 3, 1).byte().cpu().numpy()

    context_pixels = (torch.clamp(video[0, :n_context_frames], 0, 1) * 255.0).permute(0, 2, 3, 1).byte().numpy()
    return np.concatenate([context_pixels, pred_pixels], axis=0)  # (num_frames, H, W, 3)


def load_real(dataset_root: str, level_id: str, num_frames: int) -> np.ndarray:
    import av

    level_dir = os.path.join(dataset_root, level_id)
    container = av.open(os.path.join(level_dir, "rgb.mp4"))
    real_frames = []
    for i, frame in enumerate(container.decode(container.streams.video[0])):
        if i >= num_frames:
            break
        real_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(real_frames)


def main(args: Args):
    device = args.device
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    real_by_level = {lid: load_real(args.dataset_root, lid, args.num_frames) for lid in args.level_ids}
    rollouts = {lid: {} for lid in args.level_ids}

    for name, ckpt_path in [("untrained", args.untrained_ckpt), ("finetuned", args.finetuned_ckpt)]:
        worldmem, step = load_checkpoint(args.config_path, ckpt_path, device)
        print(f"[{name}] loaded checkpoint at step={step} from {ckpt_path}")
        for level_id in args.level_ids:
            rollouts[level_id][name] = generate(worldmem, args, level_id)
            print(f"  [{name}] {level_id} done")
        del worldmem
        torch.cuda.empty_cache()

    for level_id in args.level_ids:
        real = real_by_level[level_id]
        untrained, finetuned = rollouts[level_id]["untrained"], rollouts[level_id]["finetuned"]
        n = min(real.shape[0], untrained.shape[0], finetuned.shape[0])
        real_n, untrained_n, finetuned_n = real[:n], untrained[:n], finetuned[:n]
        psnr_untrained = np.array([psnr(untrained_n[t], real_n[t]) for t in range(n)], dtype=np.float32)
        psnr_finetuned = np.array([psnr(finetuned_n[t], real_n[t]) for t in range(n)], dtype=np.float32)

        out_path = os.path.join(args.output_dir, f"worldmem_{level_id}.npz")
        np.savez(
            out_path,
            real=real_n, untrained=untrained_n, finetuned=finetuned_n,
            psnr_untrained=psnr_untrained, psnr_finetuned=psnr_finetuned,
            level_id=level_id, fps=args.fps,
        )
        print(f"Saved WorldMem comparison data to {out_path} ({n} frames)")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
