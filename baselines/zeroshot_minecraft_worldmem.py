"""True zero-shot WorldMem on the SAME real Minecraft clips zeroshot_minecraft_oasis.py uses: the
ORIGINAL 25-dim checkpoint, ORIGINAL WorldMemMinecraft.interactive() call sequence (adapted
directly from eval_worldmem_zeroshot.py's own worldmem_rollout(), same init->generate pattern) --
no Craftium, no 23-dim surgery, no fine-tuning anywhere in this script.

The only change from eval_worldmem_zeroshot.py's own rollout function: instead of driving
interactive() with a synthetic key-pattern action string, this feeds it the SAME real recorded
VPT action tensor open-oasis's own sample_data provides -- both models' native action
vocabularies are the identical 25-key VPT space (see docs/action_space_comparison.md), so no
Craftium adaptation of any kind is needed here, just one index fix documented below.

Camera-index swap (required, verified): Oasis's action files (open-oasis/utils.py::ACTION_KEYS)
store cameraX at index 15, cameraY at 16. WorldMem's own generation-facing ACTION_KEYS (this
file's copy, matching eval_worldmem_zeroshot.py's and minecraft_video_dataset.py's convention,
NOT the other in-repo copy in algorithms/worldmem/models/utils.py -- see
docs/action_space_comparison.md section 1 for the full inconsistency) stores cameraY at 15,
cameraX at 16 -- the reverse. Every other index (0-14, 17-24) is identical between the two.
Without this swap, WorldMem would interpret real pitch deltas as yaw and vice versa.

Runs in WorldMem's own isolated venv (baselines/WorldMem/.venv).

Example:
    cd baselines/WorldMem
    .venv/bin/python ../zeroshot_minecraft_worldmem.py --num-frames 150 \
        --output-dir ../../outputs/zeroshot_minecraft
"""
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image

WORLDMEM_ROOT = Path(__file__).resolve().parent / "WorldMem"
sys.path.insert(0, str(WORLDMEM_ROOT))
from algorithms.worldmem import WorldMemMinecraft  # noqa: E402
from experiments.exp_base import load_custom_checkpoint  # noqa: E402

CLIPS = [
    "Player729-f153ac423f61-20210806-224813.chunk_000",
    "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001",
    "treechop-f153ac423f61-20210916-183423.chunk_000",
]

# WorldMem's generation-facing camera-index convention (index 15=cameraY, 16=cameraX) -- see
# module docstring. Oasis's own tensors use the opposite (15=cameraX, 16=cameraY); swap before use.
WORLDMEM_CAMERA_Y_IDX = 15
WORLDMEM_CAMERA_X_IDX = 16
OASIS_CAMERA_X_IDX = 15
OASIS_CAMERA_Y_IDX = 16


@dataclass
class Args:
    sample_data_dir: str = "../open-oasis/sample_data"
    clips: List[str] = field(default_factory=lambda: list(CLIPS))
    config_path: str = "configurations/huggingface.yaml"
    num_frames: int = 150
    chunk_frames: int = 50
    fps: int = 20
    output_dir: str = "../../outputs/zeroshot_minecraft"


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def load_worldmem(config_path: str, device: torch.device) -> WorldMemMinecraft:
    cfg = OmegaConf.load(config_path)  # ORIGINAL config: no action_cond_dim override
    worldmem = WorldMemMinecraft(cfg)
    load_custom_checkpoint(algo=worldmem.diffusion_model, checkpoint_path=cfg.diffusion_path)
    load_custom_checkpoint(algo=worldmem.vae, checkpoint_path=cfg.vae_path)
    load_custom_checkpoint(algo=worldmem.pose_prediction_model, checkpoint_path=cfg.pose_predictor_path)
    return worldmem.to(device).eval()


def load_image_as_tensor(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return transforms.Compose([transforms.ToTensor()])(image).numpy()


def extract_frame0_png(video_path: str, out_path: str) -> str:
    container = av.open(video_path)
    frame = next(container.decode(container.streams.video[0]))
    frame.to_image().save(out_path)
    container.close()
    return out_path


def oasis_actions_to_worldmem(oasis_actions: torch.Tensor) -> torch.Tensor:
    """Swap the camera-index pair (15<->16) to convert a real Oasis-format one-hot action
    tensor into WorldMem's own generation-facing convention. Every other column is identical."""
    swapped = oasis_actions.clone()
    swapped[:, WORLDMEM_CAMERA_Y_IDX] = oasis_actions[:, OASIS_CAMERA_Y_IDX]
    swapped[:, WORLDMEM_CAMERA_X_IDX] = oasis_actions[:, OASIS_CAMERA_X_IDX]
    return swapped


@torch.no_grad()
def worldmem_rollout(worldmem, seed_frame: np.ndarray, real_actions: torch.Tensor, total_frames: int, device, chunk_frames: int):
    """Same init->generate call sequence as eval_worldmem_zeroshot.py's worldmem_rollout(), but
    driven by real_actions (already camera-swapped, shape (total_frames-1, 25)) instead of a
    synthetic key-pattern string."""
    init_action = np.zeros(25, dtype=np.float32)
    init_pose = np.zeros(5, dtype=np.float32)

    _, mem_latents, mem_actions, mem_poses, mem_c2w, mem_idx = worldmem.interactive(
        seed_frame, init_action, init_pose, device=device,
        memory_latent_frames=None, memory_actions=None, memory_poses=None,
        memory_c2w=None, memory_frame_idx=None,
    )

    video_frames = seed_frame[None]  # (1, C, H, W), float [0, 1]
    remaining = total_frames - 1
    offset = 0
    while offset < remaining:
        n = min(chunk_frames, remaining - offset)
        input_actions = real_actions[offset : offset + n]
        new_frame, mem_latents, mem_actions, mem_poses, mem_c2w, mem_idx = worldmem.interactive(
            seed_frame, input_actions, None, device=device,
            memory_latent_frames=mem_latents, memory_actions=mem_actions, memory_poses=mem_poses,
            memory_c2w=mem_c2w, memory_frame_idx=mem_idx,
        )
        video_frames = np.concatenate([video_frames, new_frame[:, 0]])
        offset += n

    out = np.clip(video_frames, 0.0, 1.0) * 255.0
    return out  # (T, C, H, W) float [0, 255]


def read_video_frames(path: str, max_frames: int) -> np.ndarray:
    container = av.open(path)
    frames = []
    for i, frame in enumerate(container.decode(container.streams.video[0])):
        if i >= max_frames:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)  # (T, H, W, 3) uint8


def main(args: Args):
    device = torch.device("cuda")
    worldmem = load_worldmem(args.config_path, device)
    print("WorldMem original checkpoint + config loaded.")
    os.makedirs(args.output_dir, exist_ok=True)

    for clip in args.clips:
        mp4_path = os.path.join(args.sample_data_dir, f"{clip}.mp4")
        actions_path = os.path.join(args.sample_data_dir, f"{clip}.one_hot_actions.pt")

        print(f"=== Generating WorldMem native-Minecraft rollout for {clip} ===")
        oasis_actions = torch.load(actions_path, weights_only=True)[: args.num_frames - 1]  # (T-1, 25)
        real_actions = oasis_actions_to_worldmem(oasis_actions)

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_png = extract_frame0_png(mp4_path, os.path.join(tmp_dir, "frame0.png"))
            seed_frame = load_image_as_tensor(prompt_png)
            gen = worldmem_rollout(worldmem, seed_frame, real_actions, args.num_frames, device, args.chunk_frames)

        real = read_video_frames(mp4_path, args.num_frames)  # (T, H, W, 3) uint8
        gen_hwc = np.transpose(gen, (0, 2, 3, 1)).astype(np.uint8)  # (T, H, W, 3)

        if gen_hwc.shape[1:3] != real.shape[1:3]:
            # WorldMem's own native resolution may differ from the raw clip's -- resize the real
            # frames to match, matching eval_worldmem_zeroshot.py's own handling of this case.
            import torch.nn.functional as F

            real_t = torch.from_numpy(real).permute(0, 3, 1, 2).float()
            real_t = F.interpolate(real_t, size=gen_hwc.shape[1:3], mode="bilinear", align_corners=False)
            real = real_t.permute(0, 2, 3, 1).byte().numpy()

        n = min(real.shape[0], gen_hwc.shape[0])
        real_n, gen_n = real[:n], gen_hwc[:n]
        psnr_values = np.array([psnr(gen_n[t], real_n[t]) for t in range(n)], dtype=np.float32)

        out_npz = os.path.join(args.output_dir, f"worldmem_{clip}.npz")
        np.savez(out_npz, real=real_n, generated=gen_n, psnr=psnr_values, clip=clip, fps=args.fps)
        print(f"Saved {out_npz} ({n} frames, mean PSNR {psnr_values.mean():.2f}dB)")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
