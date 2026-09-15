"""Same per-episode rollout + I3D feature extraction as eval_oasis_finetuned_fvd.py, but for an
explicit SUBSET of episodes ("a shard"), and saving the raw pooled I3D features + per-frame PSNR
values to an .npz file instead of computing the final FVD itself.

Why this exists: FVD is computed ONCE from the fully pooled I3D features across every episode in
the split (see eval_oasis_finetuned_fvd.py's own `compute_frechet_distance(real_feats, gen_feats)`
call, using features concatenated across ALL episodes) -- it is NOT the average of several
per-subset FVD values (Frechet distance isn't linear across differently-sized/composed feature
pools that way). To parallelize the (expensive) generation + feature-extraction step across
multiple GPUs while still getting a result mathematically IDENTICAL to running every episode in
one process, each GPU/process runs this script on its own shard of episodes and saves its raw
features; merge_oasis_fvd_shards.py then concatenates every shard's features into one pool and
computes FVD exactly once, matching what a single big run would have produced.

Example (4-way split across 4 GPUs, run in parallel):
    cd baselines
    python eval_oasis_finetuned_fvd_shard.py --shard-index 0 --num-shards 4 \\
        --split train --ckpt-path ../outputs/finetune_oasis_fullscale/step_010000.pt \\
        --output-npz ../outputs/finetune_oasis_fullscale/shard_feats/shard_0.npz --device cuda:0
    python eval_oasis_finetuned_fvd_shard.py --shard-index 1 --num-shards 4 \\
        --split train --ckpt-path ../outputs/finetune_oasis_fullscale/step_010000.pt \\
        --output-npz ../outputs/finetune_oasis_fullscale/shard_feats/shard_1.npz --device cuda:1
    # ... shards 2, 3 on cuda:2, cuda:3 ...
    python merge_oasis_fvd_shards.py \\
        --shard-npz-glob '../outputs/finetune_oasis_fullscale/shard_feats/shard_*.npz' \\
        --output-json ../outputs/finetune_oasis_fullscale/eval_fvd_train_full.json
"""
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import av
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from finetune_oasis import load_finetuned_dit  # noqa: E402
from finetune_oasis_data import DATASET_ROOT, load_split  # noqa: E402
from finetune_oasis_rollout import generate_rollout, shift_actions  # noqa: E402

# Same load-by-file-path reasoning as eval_oasis_finetuned_fvd.py -- see that file's own comment.
import importlib.util  # noqa: E402

PERSIST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fvd_metric_spec = importlib.util.spec_from_file_location(
    "persist_fvd_metric", os.path.join(PERSIST_ROOT, "utils", "fvd_metric.py")
)
_fvd_metric = importlib.util.module_from_spec(_fvd_metric_spec)
_fvd_metric_spec.loader.exec_module(_fvd_metric)
I3D_FEATURE_DIM = _fvd_metric.I3D_FEATURE_DIM
clips_from_video = _fvd_metric.clips_from_video
extract_i3d_features = _fvd_metric.extract_i3d_features
load_i3d_model = _fvd_metric.load_i3d_model


@dataclass
class Args:
    split_path: str = "../outputs/finetune_split.json"
    split: Literal["train", "test"] = "train"
    shard_index: int = 0
    num_shards: int = 1
    """This shard handles level_ids[shard_index::num_shards] of the split -- an interleaved
    (not contiguous) slice, so shards stay balanced even if episode difficulty/length correlates
    with position in the split file."""
    ckpt_path: str = "../outputs/finetune_oasis_fullscale/step_010000.pt"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    dataset_root: str = DATASET_ROOT
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    ddim_steps: int = 10
    i3d_batch_size: int = 8
    output_npz: str = "../outputs/shard_feats/shard_0.npz"
    device: str = "cuda:0"
    seed: int = 0


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    """Verbatim copy of eval_oasis_finetuned_fvd.py's psnr() -- same formula throughout."""
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def read_full_video(path: str) -> np.ndarray:
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    container.close()
    return np.stack(frames)


@torch.no_grad()
def main(args: Args):
    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)

    model, step = load_finetuned_dit(args.ckpt_path, device)
    print(f"Loaded checkpoint at step={step} from {args.ckpt_path}")

    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    i3d = load_i3d_model(device=device)

    all_level_ids = load_split(args.split_path, args.split)
    level_ids = all_level_ids[args.shard_index :: args.num_shards]
    max_frames = max(args.frame_lengths)
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(level_ids)} of {len(all_level_ids)} "
          f"episodes from split='{args.split}', max_frames={max_frames}")

    local_real_feats = {length: [] for length in args.frame_lengths}
    local_gen_feats = {length: [] for length in args.frame_lengths}
    psnr_values = {length: [] for length in args.frame_lengths}

    t0 = time.time()
    for idx, level_id in enumerate(level_ids):
        level_dir = os.path.join(args.dataset_root, level_id)

        real_video = read_full_video(os.path.join(level_dir, "rgb.mp4"))[:max_frames]
        npz = np.load(os.path.join(level_dir, "data.npz"))
        raw_action = torch.from_numpy(npz["action"][:max_frames].astype(np.float32))
        actions = shift_actions(raw_action)

        seed_frame = torch.from_numpy(real_video[0]).float().permute(2, 0, 1) / 255.0
        gen = generate_rollout(
            model, vae, seed_frame, actions, max_frames, device,
            ddim_steps=args.ddim_steps, show_progress=False,
        )

        real = torch.from_numpy(real_video).permute(0, 3, 1, 2).float()
        gen = gen.permute(0, 3, 1, 2).float()

        n = min(gen.shape[0], real.shape[0])
        if n < max_frames:
            print(f"WARNING: {level_id}: only {n} frames available (wanted {max_frames}); truncating.")
        gen, real = gen[:n], real[:n]

        gen_np = gen.permute(0, 2, 3, 1).numpy()
        real_np = real.permute(0, 2, 3, 1).numpy()
        per_frame_psnr = [psnr(gen_np[t], real_np[t]) for t in range(gen_np.shape[0])]

        for length in args.frame_lengths:
            if gen.shape[0] < length:
                continue
            gen_clips = clips_from_video(gen[:length])
            real_clips = clips_from_video(real[:length])
            local_gen_feats[length].append(
                extract_i3d_features(gen_clips, i3d, device=device, batch_size=args.i3d_batch_size)
            )
            local_real_feats[length].append(
                extract_i3d_features(real_clips, i3d, device=device, batch_size=args.i3d_batch_size)
            )
            psnr_values[length].extend(per_frame_psnr[:length])

        elapsed = time.time() - t0
        print(f"[{idx+1}/{len(level_ids)}] {level_id} done ({elapsed:.1f}s elapsed, "
              f"{elapsed/(idx+1):.1f}s/episode avg)")

    save_dict = {"level_ids": np.array(level_ids), "frame_lengths": np.array(args.frame_lengths)}
    for length in args.frame_lengths:
        gen_feats = (
            np.concatenate(local_gen_feats[length], axis=0)
            if local_gen_feats[length] else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        real_feats = (
            np.concatenate(local_real_feats[length], axis=0)
            if local_real_feats[length] else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        save_dict[f"gen_feats_{length}"] = gen_feats
        save_dict[f"real_feats_{length}"] = real_feats
        save_dict[f"psnr_values_{length}"] = np.array(psnr_values[length], dtype=np.float32)

    np.savez(args.output_npz, **save_dict)
    elapsed = time.time() - t0
    print(f"Saved shard features to {args.output_npz} ({len(level_ids)} episodes, {elapsed:.1f}s)")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
