"""True zero-shot Oasis on its own native Minecraft data: the ORIGINAL 25-dim checkpoint, the
ORIGINAL unmodified open-oasis/generate.py code, real recorded Minecraft footage + real recorded
VPT actions -- no Craftium, no 23-dim surgery, no fine-tuning involved anywhere in this script.

Ground truth: the 3 real Minecraft clips already bundled with open-oasis
(baselines/open-oasis/sample_data/*.mp4 + matching *.one_hot_actions.pt), each a genuine
1200-frame/60s recorded session with its own paired action file. This script calls
generate.py's own main() directly (import + call, not a reimplementation) so the generation
math is guaranteed identical to the original release, then reads the produced video back plus
the real ground-truth frames to compute PSNR and save an .npz for the grid compositor.

The seed frame is passed to generate.py as an extracted PNG, not the raw .mp4, working around a
real bug in open-oasis's own (unmodified) utils.py::load_prompt: its video-loading branch never
transposes torchvision's (T,H,W,C) layout to the (C,H,W) order the function's shared resize/
rearrange logic assumes -- that logic was written against the image branch's own read_image(),
which already returns (C,H,W). Feeding a real .mp4 directly hits this and corrupts the tensor
shape before it reaches the model. Extracting frame 0 as a PNG routes through the working image
branch instead -- no changes to open-oasis's own code, and no loss of fidelity since only 1
prompt frame (n_prompt_frames=1) is ever needed here anyway.

Example:
    cd baselines
    python zeroshot_minecraft_oasis.py --num-frames 150 \
        --output-dir ../outputs/zeroshot_minecraft
"""
import argparse
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List

import av
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

CLIPS = [
    "Player729-f153ac423f61-20210806-224813.chunk_000",
    "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001",
    "treechop-f153ac423f61-20210916-183423.chunk_000",
]


@dataclass
class Args:
    sample_data_dir: str = "open-oasis/sample_data"
    clips: List[str] = field(default_factory=lambda: list(CLIPS))
    oasis_ckpt: str = "open-oasis/oasis500m.safetensors"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    num_frames: int = 150
    ddim_steps: int = 10
    fps: int = 20
    output_dir: str = "../outputs/zeroshot_minecraft"


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def read_video_frames(path: str, max_frames: int) -> np.ndarray:
    container = av.open(path)
    frames = []
    for i, frame in enumerate(container.decode(container.streams.video[0])):
        if i >= max_frames:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)


def extract_frame0_png(video_path: str, out_path: str) -> str:
    container = av.open(video_path)
    frame = next(container.decode(container.streams.video[0]))
    frame.to_image().save(out_path)
    container.close()
    return out_path


def main(args: Args):
    # Import generate.py's own main() unmodified -- this is the actual original code path,
    # not a reimplementation of its sampling loop.
    import generate as oasis_generate

    os.makedirs(args.output_dir, exist_ok=True)

    for clip in args.clips:
        mp4_path = os.path.join(args.sample_data_dir, f"{clip}.mp4")
        actions_path = os.path.join(args.sample_data_dir, f"{clip}.one_hot_actions.pt")
        out_mp4 = os.path.join(args.output_dir, f"oasis_{clip}.mp4")

        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_png = extract_frame0_png(mp4_path, os.path.join(tmp_dir, "frame0.png"))

            gen_args = argparse.Namespace(
                oasis_ckpt=args.oasis_ckpt,
                vae_ckpt=args.vae_ckpt,
                num_frames=args.num_frames,
                prompt_path=seed_png,
                actions_path=actions_path,
                video_offset=0,
                n_prompt_frames=1,
                output_path=out_mp4,
                fps=args.fps,
                ddim_steps=args.ddim_steps,
            )
            print(f"=== Generating Oasis native-Minecraft rollout for {clip} ===")
            oasis_generate.main(gen_args)

        real = read_video_frames(mp4_path, args.num_frames)  # ground truth, same clip, frame 0..N
        gen = read_video_frames(out_mp4, args.num_frames)
        n = min(real.shape[0], gen.shape[0])
        real, gen = real[:n], gen[:n]
        psnr_values = np.array([psnr(gen[t], real[t]) for t in range(n)], dtype=np.float32)

        out_npz = os.path.join(args.output_dir, f"oasis_{clip}.npz")
        np.savez(out_npz, real=real, generated=gen, psnr=psnr_values, clip=clip, fps=args.fps)
        print(f"Saved {out_npz} ({n} frames, mean PSNR {psnr_values.mean():.2f}dB)")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
