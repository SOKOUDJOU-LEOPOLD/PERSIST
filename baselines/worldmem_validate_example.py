"""Validate WorldMem's interactive() API before writing the batch adaptation.

Purpose: de-risk the batch adaptation (baselines/eval_worldmem_zeroshot.py, not yet written)
by first confirming the exact call sequence from app.py's generate() Gradio handler actually
produces a coherent rollout when run standalone, outside the Gradio UI.

Originally intended to use one of WorldMem's own bundled example images
(assets/ice_plains.png) as a known-good input, but those are Git LFS pointer files and
git-lfs isn't installable here without sudo -- so this uses a persist-eval-sample seed frame
directly instead (same 640x360 resolution their config expects), which is what we need to
validate anyway.

Mirrors app.py's setup + generate() exactly:
  1. init call: interactive(seed_frame, actions=zeros(25), first_pose=zeros(5), memory=None)
  2. generation call: interactive(seed_frame, input_actions, first_pose=None, memory=<from init>)
where input_actions comes from parse_input_to_tensor(key_string) -- a WASD-style string over
app.py's KEY_TO_ACTION mapping (Q/E=forward/back, W/S=look up/down, A/D=look left/right).
"""
import sys
from pathlib import Path

WORLDMEM_ROOT = Path(__file__).resolve().parent / "WorldMem"
sys.path.insert(0, str(WORLDMEM_ROOT))

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image

from algorithms.worldmem import WorldMemMinecraft
from experiments.exp_base import load_custom_checkpoint

torch.set_float32_matmul_precision("high")

ACTION_KEYS = [
    "inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9", "forward", "back", "left", "right",
    "cameraY", "cameraX", "jump", "sneak", "sprint", "swapHands", "attack", "use",
    "pickItem", "drop",
]
KEY_TO_ACTION = {
    "Q": ("forward", 1), "E": ("back", 1), "W": ("cameraY", -1), "S": ("cameraY", 1),
    "A": ("cameraX", -1), "D": ("cameraX", 1), "U": ("drop", 1), "N": ("noop", 1), "1": ("hotbar.1", 1),
}


def parse_input_to_tensor(input_str: str) -> torch.Tensor:
    action_tensor = torch.zeros((len(input_str), 25))
    for i, char in enumerate(input_str):
        action, value = KEY_TO_ACTION[char.upper()]
        if action in ACTION_KEYS:
            action_tensor[i, ACTION_KEYS.index(action)] = value
    return action_tensor


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return transforms.Compose([transforms.ToTensor()])(image)


def main():
    device = torch.device("cuda")
    cfg = OmegaConf.load(WORLDMEM_ROOT / "configurations" / "huggingface.yaml")
    worldmem = WorldMemMinecraft(cfg)
    load_custom_checkpoint(algo=worldmem.diffusion_model, checkpoint_path=cfg.diffusion_path)
    load_custom_checkpoint(algo=worldmem.vae, checkpoint_path=cfg.vae_path)
    load_custom_checkpoint(algo=worldmem.pose_prediction_model, checkpoint_path=cfg.pose_predictor_path)
    worldmem.to(device).eval()
    print("WorldMem model + checkpoints loaded OK.")

    seed_frame = load_image_as_tensor(
        "/home/lsokoudj/PERSIST/baselines/prompts/level_001_frame0.png"
    ).numpy()

    init_action = np.zeros((1, 25), dtype=np.float32)
    init_pose = np.zeros((1, 5), dtype=np.float32)

    # Step 1: init call -- encodes the seed frame, sets up empty memory buffers, generates nothing.
    new_frame, mem_latents, mem_actions, mem_poses, mem_c2w, mem_idx = worldmem.interactive(
        seed_frame, init_action[0], init_pose[0], device=device,
        memory_latent_frames=None, memory_actions=None, memory_poses=None,
        memory_c2w=None, memory_frame_idx=None,
    )
    print("Init call OK. memory_latent_frames shape:", mem_latents.shape)

    # Step 2: generation call -- gentle, short forward motion (a first attempt using 8x "D"
    # + 8x "Q", i.e. a hard sustained camera spin then sustained forward motion, produced
    # visibly degraded/banded output -- retrying with a much gentler input to check whether
    # that was a real bug or just an extreme, out-of-distribution action sequence).
    key_string = "Q" * 4
    input_actions = parse_input_to_tensor(key_string)
    print(f"Generating {len(key_string)} frames from key string: {key_string!r}")

    video_frames = seed_frame[None]  # (1, C, H, W)
    new_frame, mem_latents, mem_actions, mem_poses, mem_c2w, mem_idx = worldmem.interactive(
        seed_frame, input_actions, None, device=device,
        memory_latent_frames=mem_latents, memory_actions=mem_actions, memory_poses=mem_poses,
        memory_c2w=mem_c2w, memory_frame_idx=mem_idx,
    )
    video_frames = np.concatenate([video_frames, new_frame[:, 0]])
    print("Generation call OK. video_frames shape:", video_frames.shape)

    out = np.clip(video_frames.transpose(0, 2, 3, 1), 0.0, 1.0)
    out = (out * 255).astype(np.uint8)  # (T, H, W, C)

    out_dir = Path(__file__).resolve().parent / "outputs_worldmem_validate"
    out_dir.mkdir(exist_ok=True)
    for i in [0, len(out) // 2, len(out) - 1]:
        Image.fromarray(out[i]).save(out_dir / f"frame_{i:03d}.png")
    print(f"Saved sanity-check frames to {out_dir}")


if __name__ == "__main__":
    main()
