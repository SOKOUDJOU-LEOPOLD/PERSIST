"""Select the exact 115 episodes (100 train + 15 val) to fine-tune PERSIST on, from the external
percool/luanti HuggingFace dataset -- see docs/luanti-finetune-plan.md for the full plan this is
step 1 of.

Draws ONLY from episodes labeled "split": "train" in the dataset's own *.splits.json files (the
dataset's own "eval" split is reserved, never touched here). Train: 50 from motworld-s1nav-10k +
50 from motworld-hudnav-20k. Val: 8 from s1nav + 7 from hudnav (matching the 50/50 train ratio as
closely as integer division of 15 allows), drawn from the remaining "train"-labeled episodes not
already used for training. A few spare candidates are oversampled per pool as a buffer against a
chosen episode later failing preprocessing/voxelization.

Writes a manifest (dataset_toolkits/manifests/luanti_finetune_manifest.json) recording exactly
which sha256/seed were chosen and why, for reproducibility and paper citation, plus flat seed-list
text files per archive for the next step's streaming extraction.

Example:
    uv run python dataset_toolkits/select_luanti_episodes.py
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass
class Args:
    repo_id: str = "percool/luanti"
    seed: int = 0
    n_train_per_source: int = 50
    n_val_s1nav: int = 8
    n_val_hudnav: int = 7
    n_spare_train: int = 5
    n_spare_val: int = 2
    output_dir: str = "dataset_toolkits/manifests"


SOURCES = {
    "s1nav": {
        "splits_file": "motworld-s1nav-10k.splits.json",
        "archive": "motworld-s1nav-10k.tar.zst",
        "env_id": "OpenWorldCreative-v0",
    },
    "hudnav": {
        "splits_file": "motworld-hudnav-20k.splits.json",
        "archive": "motworld-hudnav-20k.tar.zst",
        "env_id": "OpenWorldCreativeHud-v0",
    },
}


def load_train_pool(repo_id: str, splits_file: str) -> list[dict]:
    path = hf_hub_download(repo_id=repo_id, filename=splits_file, repo_type="dataset")
    data = json.load(open(path))
    return [e for e in data["episodes"] if e["split"] == "train"], path


def main(args: Args):
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "repo_id": args.repo_id,
        "sources": {},
        "episodes": [],
    }

    n_val_by_source = {"s1nav": args.n_val_s1nav, "hudnav": args.n_val_hudnav}
    seed_lists: dict[str, list[str]] = {}

    for source_name, source_cfg in SOURCES.items():
        pool, splits_path = load_train_pool(args.repo_id, source_cfg["splits_file"])
        shuffled = pool.copy()
        rng.shuffle(shuffled)

        n_train = args.n_train_per_source
        n_val = n_val_by_source[source_name]
        n_spare_train = args.n_spare_train
        n_spare_val = args.n_spare_val
        total_needed = n_train + n_val + n_spare_train + n_spare_val
        assert len(shuffled) >= total_needed, (
            f"{source_name}: train pool has {len(shuffled)} episodes, need {total_needed}"
        )

        train_eps = shuffled[:n_train]
        val_eps = shuffled[n_train : n_train + n_val]
        spare_train_eps = shuffled[n_train + n_val : n_train + n_val + n_spare_train]
        spare_val_eps = shuffled[n_train + n_val + n_spare_train : total_needed]

        manifest["sources"][source_name] = {
            "splits_file": source_cfg["splits_file"],
            "splits_file_local_path": splits_path,
            "archive": source_cfg["archive"],
            "env_id": source_cfg["env_id"],
            "train_pool_size": len(pool),
            "n_train_selected": len(train_eps),
            "n_val_selected": len(val_eps),
            "n_spare_train": len(spare_train_eps),
            "n_spare_val": len(spare_val_eps),
        }

        for ep in train_eps:
            manifest["episodes"].append({**ep, "source": source_name, "env_id": source_cfg["env_id"],
                                          "split_role": "train"})
        for ep in val_eps:
            manifest["episodes"].append({**ep, "source": source_name, "env_id": source_cfg["env_id"],
                                          "split_role": "val"})
        for ep in spare_train_eps:
            manifest["episodes"].append({**ep, "source": source_name, "env_id": source_cfg["env_id"],
                                          "split_role": "spare_train"})
        for ep in spare_val_eps:
            manifest["episodes"].append({**ep, "source": source_name, "env_id": source_cfg["env_id"],
                                          "split_role": "spare_val"})

        # Flat seed list for Phase 2's streaming extraction -- train + val only (not spares,
        # extracted on demand later only if actually needed to replace a failed episode).
        seed_lists[source_name] = [e["name"] for e in train_eps + val_eps]

    manifest_path = out_dir / "luanti_finetune_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    for source_name, seeds in seed_lists.items():
        seed_list_path = out_dir / f"{source_name}_seeds.txt"
        with open(seed_list_path, "w") as f:
            f.write("\n".join(seeds) + "\n")

    val_sha256_path = out_dir / "val_sha256.txt"
    with open(val_sha256_path, "w") as f:
        f.write("\n".join(e["sha256"] for e in manifest["episodes"] if e["split_role"] == "val") + "\n")

    print(f"Manifest written to {manifest_path}")
    for source_name, info in manifest["sources"].items():
        print(f"  {source_name}: train={info['n_train_selected']} val={info['n_val_selected']} "
              f"spare_train={info['n_spare_train']} spare_val={info['n_spare_val']} "
              f"(pool size {info['train_pool_size']})")
    total_train = sum(s["n_train_selected"] for s in manifest["sources"].values())
    total_val = sum(s["n_val_selected"] for s in manifest["sources"].values())
    print(f"Total: {total_train} train, {total_val} val ({total_train + total_val} episodes to extract)")
    print(f"Seed lists: {out_dir}/{{s1nav,hudnav}}_seeds.txt")
    print(f"Val sha256 list: {val_sha256_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
