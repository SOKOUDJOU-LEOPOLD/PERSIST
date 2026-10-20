"""Quick GPU-memory sweep to find the largest safe --batch-size for generate_rollout_batch,
without waiting for full 600-frame rollouts. Peak memory is reached once the model's sliding
attention window (model.max_frames=32) fills up, since that's when per-step compute/activations
are largest -- a short rollout (a bit past 32 frames) already reaches that steady-state memory
profile, so this is a fast, cheap proxy for the real 600-frame memory ceiling.

Tries a list of candidate batch sizes in increasing order on one GPU, reports peak allocated
memory for each, and stops at the first one that OOMs (reporting everything before it as safe).

Example:
    cd baselines
    python probe_batch_size_memory.py --ckpt-path ../outputs/finetune_oasis_luanti_hudnav_1000ep/step_027500.pt \\
        --dataset-root <persist-eval-sample dir> --batch-sizes 4 8 12 16 24 32 --device cuda:0
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from eval_action_layout import list_all_levels, load_eval_actions_and_video  # noqa: E402
from finetune_oasis import load_finetuned_dit  # noqa: E402
from finetune_oasis_data import DATASET_ROOT  # noqa: E402
from finetune_oasis_rollout import generate_rollout_batch  # noqa: E402


@dataclass
class Args:
    ckpt_path: str = "../outputs/finetune_oasis_luanti_hudnav_1000ep/step_027500.pt"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    dataset_root: str = DATASET_ROOT
    use_oasis_native_action_layout: bool = True
    probe_frames: int = 64
    """Short on purpose -- just past model.max_frames=32 so the sliding-window attention context
    is already full and memory has reached its steady-state profile, without paying for a full
    600-frame rollout per batch size tried."""
    ddim_steps: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 24, 32, 48])
    device: str = "cuda:0"


@torch.no_grad()
def main(args: Args):
    device = args.device
    model, step = load_finetuned_dit(args.ckpt_path, device)
    print(f"Loaded checkpoint at step={step}")

    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    all_level_ids = list_all_levels(args.dataset_root)
    max_batch = max(args.batch_sizes)
    if len(all_level_ids) < max_batch:
        raise ValueError(f"Need at least {max_batch} episodes to test, only {len(all_level_ids)} available")

    print(f"Preloading {max_batch} episodes for reuse across all batch sizes tried...")
    loaded = [
        load_eval_actions_and_video(args.dataset_root, lid, args.probe_frames, args.use_oasis_native_action_layout)
        for lid in all_level_ids[:max_batch]
    ]
    seed_frames_all = torch.stack([seed for _, _, seed in loaded], dim=0)
    actions_all = torch.stack([actions for _, actions, _ in loaded], dim=0)

    print(f"{'batch_size':>10} | {'status':>8} | {'peak_mem_GB':>12} | {'wall_s':>8}")
    import time

    for bs in sorted(args.batch_sizes):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        seed_frames = seed_frames_all[:bs].clone()
        actions = actions_all[:bs].clone()
        t0 = time.time()
        try:
            generate_rollout_batch(
                model, vae, seed_frames, actions, args.probe_frames, device,
                ddim_steps=args.ddim_steps, show_progress=False,
            )
            elapsed = time.time() - t0
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"{bs:>10} | {'OK':>8} | {peak_gb:>12.2f} | {elapsed:>8.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>10} | {'OOM':>8} | {'--':>12} | {'--':>8}")
            print(f"Stopping sweep -- batch_size={bs} OOM'd, largest safe batch_size tested is below this.")
            break


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
