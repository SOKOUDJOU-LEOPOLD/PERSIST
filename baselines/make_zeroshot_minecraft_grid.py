"""Combines zeroshot_minecraft_oasis.py + zeroshot_minecraft_worldmem.py's per-clip .npz outputs
into one grid video, one row per real Minecraft clip, 4 columns:

    input frame (frozen seed) | GT video | WorldMem video | Oasis video

Live per-frame PSNR (vs. GT video) burned into the two generated panels. Needs only
numpy/PIL/imageio -- no model loading -- run from either venv.

Example:
    cd baselines
    python make_zeroshot_minecraft_grid.py \
        --grids-dir ../outputs/zeroshot_minecraft \
        --output-path ../outputs/zeroshot_minecraft/combined_3x4.mp4
"""
import os
from dataclasses import dataclass, field
from typing import List

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLIPS = [
    "Player729-f153ac423f61-20210806-224813.chunk_000",
    "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001",
    "treechop-f153ac423f61-20210916-183423.chunk_000",
]


@dataclass
class Args:
    grids_dir: str = "../outputs/zeroshot_minecraft"
    clips: List[str] = field(default_factory=lambda: list(CLIPS))
    output_path: str = "../outputs/zeroshot_minecraft/combined_3x4.mp4"
    fps: int = 20


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


def build_row(oasis: dict, worldmem: dict, t: int) -> np.ndarray:
    seed_frame = oasis["real"][0]
    p_input = label_panel(seed_frame, "Input frame (seed)", "")
    p_gt = label_panel(oasis["real"][t], "GT video", "")
    p_wm = label_panel(worldmem["generated"][t], "WorldMem (zero-shot)", f"PSNR: {worldmem['psnr'][t]:.1f}dB")
    p_oa = label_panel(oasis["generated"][t], "Oasis (zero-shot)", f"PSNR: {oasis['psnr'][t]:.1f}dB")
    return np.concatenate([p_input, p_gt, p_wm, p_oa], axis=1)


def main(args: Args):
    rows_data = []
    for clip in args.clips:
        oasis_path = os.path.join(args.grids_dir, f"oasis_{clip}.npz")
        worldmem_path = os.path.join(args.grids_dir, f"worldmem_{clip}.npz")
        oasis = np.load(oasis_path)
        worldmem = np.load(worldmem_path)
        rows_data.append((oasis, worldmem))

    n = min(
        min(o["real"].shape[0], o["generated"].shape[0], w["real"].shape[0], w["generated"].shape[0])
        for o, w in rows_data
    )
    print(f"{len(rows_data)} clips, {n} frames each (shortest common length)")

    frames_out = []
    for t in range(n):
        rows = [build_row(oasis, worldmem, t) for oasis, worldmem in rows_data]
        frames_out.append(np.concatenate(rows, axis=0))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    imageio.mimsave(args.output_path, frames_out, fps=args.fps)
    print(f"Saved combined {len(rows_data)}x4 zero-shot Minecraft grid to {args.output_path} ({n} frames)")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
