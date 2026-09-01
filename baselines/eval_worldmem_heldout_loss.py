"""Cheapest possible check for whether WorldMem fine-tuning did anything -- same methodology as
eval_oasis_heldout_loss.py: compare the training loss (WorldMem's own `training_step`, unmodified)
of two checkpoints on the SAME fixed batch of held-out (test-split) windows. No sampling/
generation, no I3D. `training_step` internally draws noise levels/noise via torch's global RNG
(no explicit generator plumbed through), so both checkpoints are evaluated right after an
identical `torch.manual_seed(...)` call to guarantee they see byte-identical corruption.

"Before" is step_000000.pt (pretrained backbone + freshly-initialized 23-dim action layer, saved
by finetune_worldmem.py right before training starts) -- same reasoning as the Oasis version:
the original 25-dim checkpoint can't consume Craftium's 23-dim actions at all.

Example:
    cd baselines/WorldMem
    .venv/bin/python ../eval_worldmem_heldout_loss.py \
        --before-ckpt ../../outputs/finetune_worldmem/step_000000.pt \
        --after-ckpt ../../outputs/finetune_worldmem/final.pt
"""
import os
import sys
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorldMem"))

from finetune_worldmem import CRAFTIUM_ACTION_DIM, GatedActionEmbed  # noqa: E402
from finetune_worldmem_data import CraftiumWorldMemDataset  # noqa: E402


@dataclass
class Args:
    split_path: str = "../../outputs/finetune_split.json"
    config_path: str = "configurations/huggingface.yaml"
    before_ckpt: str = "../../outputs/finetune_worldmem/step_000000.pt"
    after_ckpt: str = "../../outputs/finetune_worldmem/final.pt"
    n_frames: int = 8
    memory_condition_length: int = 8
    num_windows: int = 8
    """How many fixed held-out test-split windows to evaluate on (kept smaller than Oasis's 16
    since each WorldMem window already carries memory_condition_length extra frames)."""
    eval_seed: int = 1234
    device: str = "cuda:0"


def build_worldmem_shell(config_path: str, device: str, gated_action: bool = True):
    """Same construction as finetune_worldmem.py's load_worldmem_for_finetuning, minus the
    checkpoint loading -- that's done per-checkpoint by the caller instead."""
    from algorithms.worldmem import WorldMemMinecraft

    cfg = OmegaConf.load(config_path)
    cfg.action_cond_dim = CRAFTIUM_ACTION_DIM
    cfg.require_pose_prediction = False
    worldmem = WorldMemMinecraft(cfg)
    if gated_action:
        action_layer = worldmem.diffusion_model.model.external_cond
        gated = GatedActionEmbed(CRAFTIUM_ACTION_DIM, action_layer.out_features)
        worldmem.diffusion_model.model.external_cond = gated
    worldmem.vae.eval()
    for p in worldmem.vae.parameters():
        p.requires_grad_(False)
    return worldmem.to(device)


def main(args: Args):
    device = args.device

    dataset = CraftiumWorldMemDataset(
        args.split_path, "test", n_frames=args.n_frames, memory_condition_length=args.memory_condition_length,
    )
    print(f"Held-out test set: {len(dataset.level_ids)} episodes, {len(dataset)} total windows")

    g = torch.Generator().manual_seed(args.eval_seed)
    idxs = torch.randperm(len(dataset), generator=g)[: args.num_windows].tolist()
    batch_items = [dataset[i] for i in idxs]
    video = torch.stack([b[0] for b in batch_items]).to(device)
    actions = torch.stack([b[1] for b in batch_items]).to(device)
    poses = torch.stack([b[2] for b in batch_items]).to(device)
    timestamp = torch.stack([b[3] for b in batch_items]).to(device)
    batch = (video, actions, poses, timestamp)

    # Load the VAE (needed by every checkpoint's forward pass; unaffected by the action-layer
    # swap, so loaded once here rather than per-checkpoint like the diffusion weights).
    worldmem = build_worldmem_shell(args.config_path, device)
    from huggingface_hub import hf_hub_download

    cfg = OmegaConf.load(args.config_path)
    vae_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.vae_path.split("/")[:2]), filename="/".join(cfg.vae_path.split("/")[2:])
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    worldmem.vae.load_state_dict(vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt, strict=True)

    results = {}
    for label, ckpt_path in [("before", args.before_ckpt), ("after", args.after_ckpt)]:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        worldmem.diffusion_model.load_state_dict(ckpt["model"])
        worldmem.diffusion_model.eval()

        torch.manual_seed(args.eval_seed + 1)  # identical noise draw for both checkpoints
        with torch.no_grad():
            loss_dict = worldmem.training_step(batch, 0)
        loss = (loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict).item()
        results[label] = loss
        print(f"{label:>6} (ckpt step={ckpt.get('step')}, {ckpt_path}): held-out loss = {loss:.5f}")

    delta = results["before"] - results["after"]
    pct = 100 * delta / results["before"]
    print(f"\n=== Held-out loss comparison (for the paper) ===")
    print(f"num_windows={args.num_windows} (from {len(dataset.level_ids)} test episodes), "
          f"n_frames={args.n_frames}, memory_condition_length={args.memory_condition_length}, "
          f"eval_seed={args.eval_seed}")
    print(f"before: {results['before']:.5f}  after: {results['after']:.5f}  "
          f"delta: {delta:+.5f} ({pct:+.1f}%)")
    if delta > 0:
        print("Fine-tuning REDUCED held-out loss -- evidence it learned something.")
    else:
        print("Fine-tuning did NOT reduce held-out loss -- investigate before proceeding to "
              "qualitative/FVD checks.")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
