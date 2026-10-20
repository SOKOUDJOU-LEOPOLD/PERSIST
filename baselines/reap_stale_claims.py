"""Clears orphaned claims: a claim marker (.claims/{level_id}/) whose episode was never actually
completed (no matching {level_id}.npz), because the worker that claimed it died mid-episode --
OOM, preemption, walltime, node failure. Without this, that episode would be permanently stuck:
claim_next_batch (eval_oasis_finetuned_fvd_shard.py) treats an existing claim dir as "someone
else is already handling this" forever, so no future worker would ever retry it.

Run this BEFORE merging (run_persist256_merge.slurm does this automatically) and before any
follow-up run intended to fill gaps left by a partial/failed pool.

Example:
    cd baselines
    python reap_stale_claims.py --output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256
"""
import os
from dataclasses import dataclass


@dataclass
class Args:
    output_dir: str = "../outputs/episode_feats"
    """Same --output-dir passed to eval_oasis_finetuned_fvd_shard.py -- contains episode_feats/
    (completed .npz files) and .claims/ (claim marker dirs), both directly under this path per
    that script's own layout (output_dir/{level_id}.npz, output_dir/.claims/{level_id}/)."""


def main(args: Args):
    claims_dir = os.path.join(args.output_dir, ".claims")
    if not os.path.isdir(claims_dir):
        print(f"No claims dir at {claims_dir} -- nothing to reap.")
        return

    claimed = set(os.listdir(claims_dir))
    completed = {f[:-4] for f in os.listdir(args.output_dir) if f.endswith(".npz") and not f.startswith(".")}
    stale = claimed - completed

    for level_id in stale:
        os.rmdir(os.path.join(claims_dir, level_id))
    print(f"{len(claimed)} claimed, {len(completed)} completed, {len(stale)} stale claim(s) reaped: {sorted(stale)}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
