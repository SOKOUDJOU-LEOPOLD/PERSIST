"""Qualitative sanity check for the Oasis-native-action-layout fine-tune: generates one
comparison video -- Input Frame (frozen seed) | GT video | Oasis rollout, with live per-frame
PSNR burned into the Oasis panel -- for a single episode, mirroring generate_native_grid.py's
WorldMem version.

Builds actions via augment_actions_to_oasis_native_layout (NOT the raw 23-dim path
make_comparison_grid_oasis.py/finetune_oasis_rollout.shift_actions uses), so this uses the SAME
mapping the checkpoint was actually fine-tuned with. Reuses finetune_oasis_rollout.generate_rollout
(the same DDIM sampling function eval_oasis_finetuned_fvd.py uses) and
finetune_oasis.load_finetuned_dit, which auto-detects action_dim from the checkpoint's own weight
shape -- no need to pass action_dim explicitly.

Example:
    cd baselines
    .venv/bin/python generate_oasis_native_grid.py \
        --ckpt-path ../outputs/finetune_oasis_level_006_native_v1/final.pt \
        --level-id level_006 --device cuda:3
"""
import os
import sys
from dataclasses import dataclass

import av
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from finetune_oasis import load_finetuned_dit  # noqa: E402
from finetune_oasis_data import DATASET_ROOT  # noqa: E402
from finetune_oasis_rollout import generate_rollout, shift_actions  # noqa: E402
from craftium_action_features import augment_actions_to_oasis_native_layout  # noqa: E402


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def label_panel(frame: np.ndarray, title: str, psnr_text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    canvas = Image.new("RGB", (img.width, img.height + 36), (20, 20, 20))
    canvas.paste(img, (0, 36))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    if psnr_text:
        draw.rectangle([(0, canvas.height - 30), (220, canvas.height)], fill=(0, 0, 0))
        draw.text((8, canvas.height - 26), psnr_text, fill=(120, 255, 120), font=font)
    return np.array(canvas)


def load_real_and_native_actions(dataset_root: str, level_id: str, num_frames: int):
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
    # Mapping applied to the full 600-frame pool (camera-delta computation needs consecutive
    # full-pool frames), then sliced to num_frames -- same pattern as finetune_oasis_data.py.
    action_pool = augment_actions_to_oasis_native_layout(
        npz["action"], npz["player_yaw"], npz["player_pitch"]
    )  # (600, 25)
    raw_action = torch.from_numpy(action_pool[:num_frames].astype(np.float32))
    actions = shift_actions(raw_action)
    seed_frame = torch.from_numpy(real[0]).float().permute(2, 0, 1) / 255.0
    return real, actions, seed_frame


@dataclass
class Args:
    ckpt_path: str = "../outputs/finetune_oasis_level_006_native_v1/final.pt"
    level_id: str = "level_006"
    dataset_root: str = DATASET_ROOT
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    num_frames: int = 150
    ddim_steps: int = 10
    fps: int = 20
    output_path: str = "../outputs/finetune_oasis_level_006_native_v1/grid_level_006.mp4"
    device: str = "cuda:3"
    seed: int = 0


@torch.no_grad()
def main(args: Args):
    device = args.device
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    model, step = load_finetuned_dit(args.ckpt_path, device)
    print(f"Loaded checkpoint at step={step} from {args.ckpt_path}")

    real, actions, seed_frame = load_real_and_native_actions(args.dataset_root, args.level_id, args.num_frames)
    gen = generate_rollout(
        model, vae, seed_frame, actions, args.num_frames, device, ddim_steps=args.ddim_steps,
    ).numpy()  # (T, H, W, 3) uint8

    n = min(real.shape[0], gen.shape[0])
    real_n, gen_n = real[:n], gen[:n]
    psnr_per_frame = np.array([psnr(gen_n[t], real_n[t]) for t in range(n)], dtype=np.float32)
    print(f"Rollout: {n} frames, PSNR mean={psnr_per_frame.mean():.2f}dB "
          f"min={psnr_per_frame.min():.2f}dB max={psnr_per_frame.max():.2f}dB")

    seed_panel = real_n[0]
    frames_out = []
    for t in range(n):
        p_input = label_panel(seed_panel, "Input Frame (seed)", "")
        p_gt = label_panel(real_n[t], "GT video", "")
        p_oasis = label_panel(gen_n[t], "Oasis (native layout)", f"PSNR: {psnr_per_frame[t]:.1f}dB")
        frames_out.append(np.concatenate([p_input, p_gt, p_oasis], axis=1))

    imageio.mimsave(args.output_path, frames_out, fps=args.fps)
    print(f"Saved grid video to {args.output_path} ({n} frames)")

    npz_path = args.output_path.replace(".mp4", ".npz")
    np.savez(npz_path, real=real_n, generated=gen_n, psnr=psnr_per_frame, level_id=args.level_id, fps=args.fps)
    print(f"Saved raw arrays to {npz_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
