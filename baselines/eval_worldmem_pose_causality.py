"""Real/shuffled/zero POSE causality ablation for WorldMem, re-run after the radius=30->1.5
Craftium recalibration (see df_video.py::_generate_condition_indices comment). Actions are held
real and fixed throughout; only the pose tensor fed to validation_step is varied, isolating pose's
own causal effect on generation quality (same methodology as the earlier gate-fix ablation).

IMPORTANT CAVEAT: the checkpoint being tested was fine-tuned under the OLD radius=30 memory
selection (finetune_worldmem_data.py's _select_memory_indices, used during training-data
construction). This script only changes the radius at INFERENCE time (df_video.py, used by
validation_step), so it is a cheap probe of the inference-side effect alone -- not a full
re-validation with re-training under the new radius too. Treat a positive result here as
justification to re-fine-tune with the fix applied at training time as well, not as the final word.

Example:
    cd baselines/WorldMem
    .venv/bin/python ../eval_worldmem_pose_causality.py \
        --ckpt-path ../../outputs/overfit_worldmem_level_006_lr5e5/step_011000.pt \
        --level-id level_006 --num-frames 150
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorldMem"))

from finetune_worldmem_generate import build_worldmem_shell, load_single_episode_batch  # noqa: E402
from finetune_worldmem_data import DATASET_ROOT  # noqa: E402


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def to_np(x):
    x = torch.clamp(x[:, 0], 0, 1) * 255.0
    return x.permute(0, 2, 3, 1).byte().cpu().numpy()


def main():
    import tyro
    from dataclasses import dataclass

    @dataclass
    class Args:
        ckpt_path: str = "../../outputs/overfit_worldmem_level_006_lr5e5/step_011000.pt"
        config_path: str = "configurations/huggingface.yaml"
        level_id: str = "level_006"
        dataset_root: str = DATASET_ROOT
        num_frames: int = 150
        memory_condition_length: int = 8
        device: str = "cuda:0"
        seed: int = 0

    args = tyro.cli(Args)
    device = args.device
    torch.manual_seed(args.seed)

    worldmem = build_worldmem_shell(args.config_path, device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    worldmem.diffusion_model.load_state_dict(ckpt["model"])
    worldmem.diffusion_model.eval()
    print(f"Loaded checkpoint at step={ckpt.get('step')} from {args.ckpt_path}")

    from huggingface_hub import hf_hub_download
    cfg = OmegaConf.load(args.config_path)
    vae_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.vae_path.split("/")[:2]), filename="/".join(cfg.vae_path.split("/")[2:])
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    worldmem.vae.load_state_dict(vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt, strict=True)

    video, actions, poses, timestamp = load_single_episode_batch(
        args.dataset_root, args.level_id, args.num_frames, args.memory_condition_length, args.seed
    )
    video, actions, poses, timestamp = (
        video.to(device), actions.to(device), poses.to(device), timestamp.to(device)
    )

    rng = torch.Generator().manual_seed(args.seed)
    n_frames = poses.shape[1]
    perm = torch.randperm(n_frames, generator=rng)
    poses_shuffled = poses[:, perm].clone()
    poses_zero = torch.zeros_like(poses)

    results = {}
    for name, pose_variant in [("real", poses), ("shuffled", poses_shuffled), ("zero", poses_zero)]:
        batch = (video, actions, pose_variant, timestamp)
        worldmem.validation_step_outputs = []
        with torch.no_grad():
            worldmem.validation_step(batch, 0)
        xs_pred, xs_decode = worldmem.validation_step_outputs[-1]
        gen_np, real_np = to_np(xs_pred), to_np(xs_decode)
        n = min(gen_np.shape[0], real_np.shape[0])
        per_frame = [psnr(gen_np[t], real_np[t]) for t in range(n)]
        results[name] = float(np.mean(per_frame))
        print(f"{name:>10s} pose -> PSNR: {results[name]:.2f}dB")

    print()
    print(f"Radius-1.5 (Craftium-recalibrated) causality result on {args.level_id}:")
    print(f"  real={results['real']:.2f}dB  shuffled={results['shuffled']:.2f}dB  zero={results['zero']:.2f}dB")
    gap_shuffle = results["real"] - results["shuffled"]
    gap_zero = results["real"] - results["zero"]
    print(f"  real-shuffled gap: {gap_shuffle:+.2f}dB   real-zero gap: {gap_zero:+.2f}dB")
    if results["real"] > results["shuffled"] > results["zero"]:
        print("  ORDERING CORRECT: real > shuffled > zero -- genuine causal pose signal.")
    else:
        print("  ORDERING NOT CLEAN -- pose still not cleanly causal at this radius.")


if __name__ == "__main__":
    main()
