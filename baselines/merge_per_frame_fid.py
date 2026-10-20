"""Stage 2 (CPU-bound): loads every episode's per-frame FID features (from
extract_per_frame_fid_features.py) and computes ONE Frechet distance per frame index -- pooling
ONLY that exact frame across all episodes (strict per-frame, no windowing/temporal smoothing).

This is scipy.linalg.sqrtm on a real x gen covariance matrix, once per frame index -- CPU-only,
does not use the GPU at all, so it's parallelized across CPU cores (multiprocessing.Pool) rather
than GPUs. Sample size per frame is num_episodes (e.g. 256 or 200) -- well below the 2048-dim
feature space, so the covariance matrix is rank-deficient/singular at every frame index
(scipy will warn "Matrix is singular... result might be inaccurate"). That's expected and
accepted here -- this is meant to show the SHAPE of drift over frames, not a rigorously accurate
absolute FID value (the already-computed FID@200/400/600 numbers, pooling far more samples, are
the trustworthy absolute numbers).

Example:
    cd baselines
    python merge_per_frame_fid.py \\
        --feats-dir ../outputs/.../per_frame_fid_feats \\
        --num-frames 600 --num-workers 48 \\
        --output-csv ../outputs/.../fid_per_frame.csv
"""
import glob
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

PERSIST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fvd_metric_spec = importlib.util.spec_from_file_location(
    "persist_fvd_metric", os.path.join(PERSIST_ROOT, "utils", "fvd_metric.py")
)
_fvd_metric = importlib.util.module_from_spec(_fvd_metric_spec)
_fvd_metric_spec.loader.exec_module(_fvd_metric)
compute_frechet_distance = _fvd_metric.compute_frechet_distance


@dataclass
class Args:
    feats_dir: str = "per_frame_fid_feats"
    num_frames: int = 600
    num_workers: int = 48
    output_csv: str = "fid_per_frame.csv"


# Set by _init_worker in each pool process -- avoids pickling the full (episodes, frames, 2048)
# arrays through IPC for every one of the 600 tasks; each worker process gets its own copy once.
_REAL = None
_GEN = None


def _init_worker(real_feats, gen_feats):
    global _REAL, _GEN
    _REAL = real_feats
    _GEN = gen_feats


def _fid_at_frame(t):
    return t, compute_frechet_distance(_REAL[:, t, :], _GEN[:, t, :])


def main(args: Args):
    paths = sorted(glob.glob(os.path.join(args.feats_dir, "*.npz")))
    if not paths:
        raise ValueError(f"No feature files found in {args.feats_dir}")
    print(f"Loading {len(paths)} episodes' features...")

    sample = np.load(paths[0])
    feature_dim = sample["real_feats"].shape[-1]
    real_feats = np.zeros((len(paths), args.num_frames, feature_dim), dtype=np.float32)
    gen_feats = np.zeros((len(paths), args.num_frames, feature_dim), dtype=np.float32)
    for i, path in enumerate(paths):
        d = np.load(path)
        n = int(d["num_frames"])
        real_feats[i, :n] = d["real_feats"][:n]
        gen_feats[i, :n] = d["gen_feats"][:n]

    print(f"Computing per-frame FID for {args.num_frames} frames across {args.num_workers} CPU workers...")
    results = [None] * args.num_frames
    with Pool(args.num_workers, initializer=_init_worker, initargs=(real_feats, gen_feats)) as pool:
        for t, fid_t in pool.imap_unordered(_fid_at_frame, range(args.num_frames)):
            results[t] = fid_t
            if t % 50 == 0:
                print(f"frame {t}: FID={fid_t:.4f}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        f.write("frame,fid\n")
        for t, fid_t in enumerate(results):
            f.write(f"{t},{fid_t:.6f}\n")

    print(f"Saved per-frame FID ({args.num_frames} points, {len(paths)} episodes pooled per point) to {args.output_csv}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
