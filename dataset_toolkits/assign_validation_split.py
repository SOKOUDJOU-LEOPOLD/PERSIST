"""Assign an exact, specific validation split to a dataset's metadata.csv, replacing
build_dataset.sh's default `--set_validation_samples vae` (which picks a random 5% tail-slice).

Used in place of that step when the fine-tuning task requires a specific, pre-selected set of
validation episodes (e.g. from dataset_toolkits/select_luanti_episodes.py's manifest) rather than
an arbitrary random split -- see docs/luanti-finetune-plan.md, Phase 5.3.

Fails loudly (does not silently substitute) if any requested validation sha256 did not survive
preprocessing/voxelization -- the correct remediation is to pick a different episode from the
manifest's spare pool and re-run, not to silently drop/replace it here.

Example:
    uv run python dataset_toolkits/assign_validation_split.py \
        --dataset_dir datasets/luanti_finetune \
        --val_sha256_file dataset_toolkits/manifests/val_sha256.txt
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Args:
    dataset_dir: str
    val_sha256_file: str
    replace_rejected: bool = False
    """Instead of failing, replace each validation sha256 that did not survive preprocessing with a random
    surviving non-validation episode from the same source (same env dir), so the split keeps its size and
    per-source balance. The replacement moves out of training. Replacements are written next to
    val_sha256_file as *.replacements.csv. For large splits where a few rejections are expected."""
    seed: int = 0


def replace_rejected_val(metadata, val_sha256, seed, out_path):
    import random

    usable = metadata["voxelized"] & (metadata["level_quality_score"] > 0)
    source = metadata["local_path"].str.split("/").str[1]  # raw/<env_id>/<seed>
    rejected = [s for s in val_sha256 if s not in metadata.index or not usable[s]]
    rng = random.Random(seed)
    val_set, replacements = set(val_sha256), []
    for s in rejected:
        src = source[s] if s in metadata.index else None
        pool = sorted(metadata.index[usable & (source == src) & ~metadata.index.isin(val_set)])
        assert pool, f"no surviving episode left to replace rejected validation sha256 {s} (source {src})"
        new = rng.choice(pool)
        val_set.discard(s)
        val_set.add(new)
        replacements.append({"rejected": s, "source": src, "replacement": new})
    pd.DataFrame(replacements, columns=["rejected", "source", "replacement"]).to_csv(out_path, index=False)
    print(f"Replaced {len(replacements)} rejected validation episodes (see {out_path})")
    return [r for r in val_sha256 if r in val_set] + [r["replacement"] for r in replacements]


def main(args: Args):
    metadata_path = f"{args.dataset_dir}/metadata.csv"
    metadata = pd.read_csv(metadata_path, dtype={"sha256": str}).set_index("sha256")

    val_sha256 = [line.strip() for line in open(args.val_sha256_file) if line.strip()]
    if args.replace_rejected:
        val_sha256 = replace_rejected_val(metadata, val_sha256, args.seed,
                                          args.val_sha256_file.removesuffix(".txt") + ".replacements.csv")

    missing = [s for s in val_sha256 if s not in metadata.index]
    assert not missing, (
        f"{len(missing)} validation sha256(s) not found in metadata.csv at all (never extracted "
        f"or never enumerated): {missing}"
    )

    not_voxelized = [s for s in val_sha256 if not metadata.loc[s, "voxelized"]]
    assert not not_voxelized, (
        f"{len(not_voxelized)} validation sha256(s) found but not voxelized (failed "
        f"preprocessing/validity checks): {not_voxelized} -- replace these from the manifest's "
        f"spare_val pool and re-run, do not silently drop them here."
    )

    bad_quality = [s for s in val_sha256 if not (metadata.loc[s, "level_quality_score"] > 0)]
    assert not bad_quality, (
        f"{len(bad_quality)} validation sha256(s) have level_quality_score <= 0: {bad_quality}"
    )

    metadata["validation_set"] = False
    metadata.loc[val_sha256, "validation_set"] = True

    metadata.to_csv(metadata_path)

    n_val = int(metadata["validation_set"].sum())
    n_train = int((~metadata["validation_set"]).sum())
    print(f"Assigned validation_set: {n_val} True / {n_train} False (total {len(metadata)})")
    assert n_val == len(val_sha256), f"expected {len(val_sha256)} val rows, got {n_val}"


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
