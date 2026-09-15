"""Combines make_comparison_grid_oasis.py + make_comparison_grid_worldmem.py's per-episode .npz
outputs (looked up as <grids_dir>/oasis_<level_id>.npz / worldmem_<level_id>.npz) into one grid
video, one row per episode, 6 columns each:

    GT input (frozen seed frame) | GT video | WorldMem untrained | WorldMem fine-tuned | Oasis untrained | Oasis fine-tuned

Live per-frame PSNR (vs. GT video) is burned into each of the 4 generated panels. Needs only
numpy/PIL/imageio -- no model loading -- so it runs in either venv (or the base PERSIST one).

Example:
    cd baselines
    python combine_comparison_grids.py \
        --grids-dir ../outputs/qualitative_grids \
        --level-ids level_001,level_003,level_004,level_005,level_002,level_040,level_060,level_077 \
        --output-path ../outputs/qualitative_grids/combined_8x6.mp4
"""
import os
from dataclasses import dataclass, field
from typing import List

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Args:
    grids_dir: str = "../outputs/qualitative_grids"
    level_ids: List[str] = field(default_factory=list)
    output_path: str = "../outputs/qualitative_grids/combined_8x6.mp4"
    fps: int = 20


def label_panel(frame: np.ndarray, title: str, psnr_text: str) -> np.ndarray:
    """Adds a title bar on top and a live per-frame PSNR readout at the bottom-left."""
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
    p_gt_input = label_panel(seed_frame, "GT input (seed frame)", "")
    p_gt_video = label_panel(oasis["real"][t], "GT video", "")
    p_wm_un = label_panel(
        worldmem["untrained"][t], "WorldMem (untrained)", f"PSNR: {worldmem['psnr_untrained'][t]:.1f}dB"
    )
    p_wm_ft = label_panel(
        worldmem["finetuned"][t], "WorldMem (fine-tuned)", f"PSNR: {worldmem['psnr_finetuned'][t]:.1f}dB"
    )
    p_oa_un = label_panel(
        oasis["untrained"][t], "Oasis (untrained)", f"PSNR: {oasis['psnr_untrained'][t]:.1f}dB"
    )
    p_oa_ft = label_panel(
        oasis["finetuned"][t], "Oasis (fine-tuned)", f"PSNR: {oasis['psnr_finetuned'][t]:.1f}dB"
    )
    return np.concatenate([p_gt_input, p_gt_video, p_wm_un, p_wm_ft, p_oa_un, p_oa_ft], axis=1)


def main(args: Args):
    if not args.level_ids:
        raise ValueError("--level-ids must list the episodes to include, one row each.")

    rows_data = []
    for level_id in args.level_ids:
        oasis_path = os.path.join(args.grids_dir, f"oasis_{level_id}.npz")
        worldmem_path = os.path.join(args.grids_dir, f"worldmem_{level_id}.npz")
        oasis = np.load(oasis_path)
        worldmem = np.load(worldmem_path)
        if str(oasis["level_id"]) != level_id or str(worldmem["level_id"]) != level_id:
            raise ValueError(f"level_id mismatch loading {level_id} -- check the npz files' own level_id field.")
        rows_data.append((oasis, worldmem))

    n = min(
        min(o["real"].shape[0], o["untrained"].shape[0], o["finetuned"].shape[0],
            w["real"].shape[0], w["untrained"].shape[0], w["finetuned"].shape[0])
        for o, w in rows_data
    )
    print(f"{len(rows_data)} episodes, {n} frames each (shortest common length)")

    frames_out = []
    for t in range(n):
        rows = [build_row(oasis, worldmem, t) for oasis, worldmem in rows_data]
        frames_out.append(np.concatenate(rows, axis=0))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    imageio.mimsave(args.output_path, frames_out, fps=args.fps)
    print(f"Saved combined {len(rows_data)}x6 comparison grid to {args.output_path} ({n} frames)")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
