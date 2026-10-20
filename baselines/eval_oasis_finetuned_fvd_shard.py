"""Dynamic work-queue worker for FVD/FID/PSNR + video generation on the full persist-eval-sample
(or any all_episodes) set, or a fixed train/test split. Unlike a static shard (a fixed, pre-decided
slice of episodes per process), this is a WORKER: it repeatedly claims the next unclaimed episode(s)
from a shared pool (one atomic `os.makedirs` per episode as the claim -- succeeds for exactly one
worker, fails for everyone else, no locking library needed on a POSIX/Lustre filesystem) and
processes them, until none remain. Any number of workers -- any GPU type, started at any time, as
GPUs free up -- can point at the SAME pool concurrently: fast GPUs naturally claim and finish more
episodes than slow ones, with no need to predict relative speed or split ranges upfront. This is
what actually gets you "use whichever GPU frees up next, of whatever type" instead of a fixed plan.

Each worker saves ONE small .npz per episode it completes (not one big file per shard) --
merge_oasis_fvd_shards.py globs the whole output directory and pools everything the same way
regardless of how many workers of what type actually ran. FVD/FID must be computed ONCE from the
fully pooled features across every episode (not averaged per-worker/per-shard -- Frechet distance
isn't linear that way), which is exactly what the merge step does.

Episodes are still generated in BATCHES within each worker (generate_rollout_batch, --batch-size)
-- generation is autoregressive (no parallelism across time within one episode), so batching
several claimed episodes into one DDIM loop is the real per-GPU throughput lever; the work-queue
above is what gives you parallelism ACROSS GPUs/nodes/types.

Example -- launch this as many times as you want, any GPU type, any time (see
submit_persist256_eval.sh for the full multi-pool submission):
    cd baselines
    python eval_oasis_finetuned_fvd_shard.py --all-episodes \\
        --dataset-root <persist-eval-sample snapshot dir> \\
        --ckpt-path ../outputs/finetune_oasis_luanti_hudnav_1000ep/step_027500.pt \\
        --use-oasis-native-action-layout --batch-size 4 \\
        --video-output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256/videos \\
        --output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256/episode_feats \\
        --device cuda:0
"""
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from eval_action_layout import list_all_levels, load_eval_actions_and_video  # noqa: E402
from finetune_oasis import load_finetuned_dit  # noqa: E402
from finetune_oasis_data import DATASET_ROOT, load_split  # noqa: E402
from finetune_oasis_rollout import generate_rollout_batch  # noqa: E402

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

_fid_metric_spec = importlib.util.spec_from_file_location(
    "persist_fid_metric", os.path.join(PERSIST_ROOT, "utils", "fid_metric.py")
)
_fid_metric = importlib.util.module_from_spec(_fid_metric_spec)
_fid_metric_spec.loader.exec_module(_fid_metric)
FID_FEATURE_DIM = _fid_metric.FID_FEATURE_DIM
extract_fid_features = _fid_metric.extract_fid_features
load_fid_inception = _fid_metric.load_fid_inception


@dataclass
class Args:
    split_path: str = "../outputs/finetune_split.json"
    split: str = "train"
    """Any key present in the split_path JSON -- e.g. "train"/"test" for the Luanti-specific
    split files, or "eval" for motworld-hudnav-20k's own held-out-episode labeling. Not
    restricted to a fixed Literal set since different split files use different label names."""
    all_episodes: bool = False
    """When True, ignore split_path/split and evaluate every level_* dir directly under
    dataset_root (for eval sets like persist-eval-sample that have no train/test split json)."""
    ckpt_path: str = "../outputs/finetune_oasis_fullscale/step_010000.pt"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    dataset_root: str = DATASET_ROOT
    use_oasis_native_action_layout: bool = False
    """Set True for a checkpoint trained with --use-oasis-native-action-layout (25-dim actions,
    e.g. the Luanti hudnav fine-tune) -- False (raw 23-dim) for older non-native checkpoints."""
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    ddim_steps: int = 10
    batch_size: int = 4
    """Episodes claimed and generated together per generate_rollout_batch call -- the per-GPU
    throughput lever (see module docstring). Tune down if this OOMs on a given GPU/frame-length."""
    i3d_batch_size: int = 8
    fid_batch_size: int = 32
    video_output_dir: str = ""
    """When non-empty, saves each episode's generated max(frame_lengths)-frame rollout as
    {video_output_dir}/{level_id}.mp4 -- the SAME rollout used for FVD/FID scoring."""
    video_fps: int = 24
    output_dir: str = "../outputs/episode_feats"
    """One small {level_id}.npz written per completed episode (not one file per worker) -- so
    however many workers of whatever GPU type actually ran, merge_oasis_fvd_shards.py just globs
    this directory. Also doubles as the claim/completion record: an episode with a finished
    .npz here is skipped by any worker (including a fresh one started later)."""
    max_episodes: int = 0
    """If >0, this worker stops after successfully completing this many episodes (0 = keep
    claiming until the whole pool is exhausted). Useful for a worker with a short SLURM time
    limit that should do a bounded amount of work and exit cleanly rather than risk being killed
    mid-batch."""
    device: str = "cuda:0"
    seed: int = 0


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    """Verbatim copy of eval_oasis_finetuned_fvd.py's psnr() -- same formula throughout."""
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def claim_next_batch(all_level_ids: List[str], output_dir: str, claims_dir: str, batch_size: int) -> List[str]:
    """Scans all_level_ids in order and atomically claims up to batch_size not-yet-done,
    not-yet-claimed episodes. os.makedirs(..., exist_ok=False) is an atomic claim primitive on a
    POSIX/Lustre filesystem -- it succeeds for exactly one racing worker and raises FileExistsError
    for every other one, so concurrent workers never double-claim the same episode without needing
    a separate locking library. Returns fewer than batch_size (possibly zero) once the pool nears
    exhaustion."""
    claimed = []
    for level_id in all_level_ids:
        if len(claimed) >= batch_size:
            break
        if os.path.exists(os.path.join(output_dir, f"{level_id}.npz")):
            continue  # already completed (by this worker in a prior batch, or another worker)
        try:
            os.makedirs(os.path.join(claims_dir, level_id), exist_ok=False)
        except FileExistsError:
            continue  # another worker claimed it first
        claimed.append(level_id)
    return claimed


@torch.no_grad()
def main(args: Args):
    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    claims_dir = os.path.join(args.output_dir, ".claims")
    os.makedirs(claims_dir, exist_ok=True)
    if args.video_output_dir:
        os.makedirs(args.video_output_dir, exist_ok=True)

    model, step = load_finetuned_dit(args.ckpt_path, device)
    print(f"Loaded checkpoint at step={step} from {args.ckpt_path}")

    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    i3d = load_i3d_model(device=device)
    fid_inception = load_fid_inception(device=device)

    all_level_ids = list_all_levels(args.dataset_root) if args.all_episodes else load_split(args.split_path, args.split)
    max_frames = max(args.frame_lengths)
    print(f"Worker pool size {len(all_level_ids)} episodes, max_frames={max_frames}, "
          f"batch_size={args.batch_size}, use_oasis_native_action_layout={args.use_oasis_native_action_layout}, "
          f"output_dir={args.output_dir}")

    t0 = time.time()
    num_done = 0
    while args.max_episodes <= 0 or num_done < args.max_episodes:
        remaining_budget = args.batch_size if args.max_episodes <= 0 else min(args.batch_size, args.max_episodes - num_done)
        batch_ids = claim_next_batch(all_level_ids, args.output_dir, claims_dir, remaining_budget)
        if not batch_ids:
            print("No unclaimed episodes remain -- pool exhausted, exiting.")
            break

        loaded = [
            load_eval_actions_and_video(args.dataset_root, lid, max_frames, args.use_oasis_native_action_layout)
            for lid in batch_ids
        ]
        avail_lens = [real.shape[0] for real, _, _ in loaded]
        action_dim = loaded[0][1].shape[-1]
        actions_batch = torch.zeros(len(loaded), max_frames, action_dim, dtype=torch.float32)
        seed_frames_batch = torch.stack([seed for _, _, seed in loaded], dim=0)
        for i, (_, actions, _) in enumerate(loaded):
            actions_batch[i, : actions.shape[0]] = actions

        gen_batch = generate_rollout_batch(
            model, vae, seed_frames_batch, actions_batch, max_frames, device,
            ddim_steps=args.ddim_steps, show_progress=True,
            desc=f"batch[{','.join(batch_ids)}]",
        )  # (B, max_frames, H, W, 3) uint8, CPU

        for i, level_id in enumerate(batch_ids):
            real_video = loaded[i][0]
            n = min(avail_lens[i], gen_batch.shape[1])
            if n < max_frames:
                print(f"WARNING: {level_id}: only {n} frames available (wanted {max_frames}); truncating.")
            gen_np = gen_batch[i, :n].numpy()
            real_np = real_video[:n]

            if args.video_output_dir:
                imageio.mimsave(os.path.join(args.video_output_dir, f"{level_id}.mp4"), gen_np, fps=args.video_fps)

            gen_chw = torch.from_numpy(gen_np).permute(0, 3, 1, 2).float()
            real_chw = torch.from_numpy(real_np).permute(0, 3, 1, 2).float()
            per_frame_psnr = [psnr(gen_np[t], real_np[t]) for t in range(n)]

            save_dict = {"level_id": level_id}
            for length in args.frame_lengths:
                if n < length:
                    save_dict[f"gen_feats_{length}"] = np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
                    save_dict[f"real_feats_{length}"] = np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
                    save_dict[f"gen_fid_feats_{length}"] = np.zeros((0, FID_FEATURE_DIM), dtype=np.float32)
                    save_dict[f"real_fid_feats_{length}"] = np.zeros((0, FID_FEATURE_DIM), dtype=np.float32)
                    save_dict[f"psnr_values_{length}"] = np.zeros((0,), dtype=np.float32)
                    continue
                gen_clips = clips_from_video(gen_chw[:length])
                real_clips = clips_from_video(real_chw[:length])
                save_dict[f"gen_feats_{length}"] = extract_i3d_features(gen_clips, i3d, device=device, batch_size=args.i3d_batch_size)
                save_dict[f"real_feats_{length}"] = extract_i3d_features(real_clips, i3d, device=device, batch_size=args.i3d_batch_size)
                save_dict[f"gen_fid_feats_{length}"] = extract_fid_features(gen_chw[:length].byte(), fid_inception, device=device, batch_size=args.fid_batch_size)
                save_dict[f"real_fid_feats_{length}"] = extract_fid_features(real_chw[:length].byte(), fid_inception, device=device, batch_size=args.fid_batch_size)
                save_dict[f"psnr_values_{length}"] = np.array(per_frame_psnr[:length], dtype=np.float32)

            # Write to a temp name then rename (atomic on the same filesystem) so a worker killed
            # mid-write never leaves a half-written .npz that a later "is it done" check would
            # wrongly treat as complete. tmp_path must already end in ".npz" -- np.savez silently
            # APPENDS ".npz" to whatever name it's given if it doesn't already end with that
            # extension, so a name like "X.npz.tmp" actually gets written as "X.npz.tmp.npz",
            # and the rename below would then look for the wrong filename.
            tmp_path = os.path.join(args.output_dir, f".{level_id}.tmp.npz")
            final_path = os.path.join(args.output_dir, f"{level_id}.npz")
            np.savez(tmp_path, **save_dict)
            os.rename(tmp_path, final_path)

            num_done += 1
            elapsed = time.time() - t0
            print(f"[{num_done} done] {level_id} ({elapsed:.1f}s elapsed, {elapsed/num_done:.1f}s/episode avg)")

    elapsed = time.time() - t0
    print(f"Worker finished: {num_done} episodes completed in {elapsed:.1f}s")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
