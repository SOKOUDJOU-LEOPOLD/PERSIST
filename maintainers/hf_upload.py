"""Publish PERSIST artifacts to the HuggingFace Hub (maintainers only).

Two subcommands:

  * ``upload-eval-dataset`` — pack the minimal files needed to run inference for a few
    validation episodes out of a full local dataset and push them as an HF *dataset* repo.
    Only the files the eval loader (`MinetestLatentCameraActionEval`, which sets
    ``no_latents=True``) actually reads are included: ``metadata.csv`` (validation rows
    only), ``dataset_params.json``, ``raw/<env>/<seed>/{data.npz,rgb.mp4}`` and
    ``voxel_classes/<sha256>.npz``. Latents are deliberately excluded to keep it small.

  * ``upload-weights`` — push the trained checkpoints as one HF *model* repo each, named
    ``<namespace>/<prefix><repo-suffix>`` (default prefix ``persist-``), with the weight stored as a
    single ``model.safetensors``. The voxel denoiser is variant-specific (``...-voxel-denoiser-s``
    and ``...-voxel-denoiser-xl``); all other models are shared. This matches the layout
    ``scripts.run_inference`` expects from ``--checkpoints-namespace`` + ``--pipeline-variant``.

Prerequisite: ``huggingface-cli login`` with a write token whose user is a member of the
target org (e.g. PERSIST-team) with write access.

Examples:
    python -m dataset_toolkits.hf_upload upload-eval-dataset \
        --src datasets/dynamic48_L8 --repo-id PERSIST-team/persist-eval-sample

    python -m dataset_toolkits.hf_upload upload-weights \
        --namespace PERSIST-team \
        --voxel-encoder ckpts/voxel_vae/model.safetensors \
        --voxel-decoder ckpts/voxel_vae/model_1.safetensors \
        --pixel-vae ckpts/pixel_vae/model.safetensors \
        --camera-model ckpts/camera_model/model.safetensors \
        --voxel-denoiser-s ckpts/voxel_dit_s/model.safetensors \
        --voxel-denoiser-xl ckpts/voxel_dit_xl/model.safetensors \
        --pixel-denoiser ckpts/pixel_dit/model.safetensors \
        --voxel-class-embedder ckpts/pixel_dit/model_1.safetensors
"""
import os
import shutil

import pandas as pd
import tyro
from loguru import logger


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def upload_eval_dataset(
    src: str,
    repo_id: str,
    num_episodes: int = 2,
    staging_dir: str = "outputs/hf_eval_sample",
    private: bool = False,
    create: bool = True,
) -> None:
    """Pack ``num_episodes`` validation episodes from ``src`` and upload them as a dataset repo.

    Args:
        src: Full local dataset root (e.g. ``datasets/dynamic48_L8``).
        repo_id: Target HF dataset repo id (e.g. ``PERSIST-team/persist-eval-sample``).
        num_episodes: How many validation episodes to include.
        staging_dir: Local folder assembled before upload (overwritten).
        private: Create the repo private instead of public.
        create: Create the repo (``exist_ok=True``) before uploading.
    """
    from huggingface_hub import HfApi

    # Filter to validation episodes via pandas — the metadata has an embedded-dict column
    # ("level_validity_conditions") whose CSV quoting a naive line copy would corrupt.
    metadata = pd.read_csv(os.path.join(src, "metadata.csv"))
    val = metadata[metadata["validation_set"] == True]  # noqa: E712
    if len(val) == 0:
        raise ValueError(f"No validation episodes (validation_set==True) found in {src}/metadata.csv.")
    selected = val.head(num_episodes)
    logger.info(f"Selected {len(selected)}/{len(val)} validation episode(s) from {src}.")

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # dataset-level config the loader reads
    _copy(os.path.join(src, "dataset_params.json"), os.path.join(staging_dir, "dataset_params.json"))

    # per-episode files
    for _, row in selected.iterrows():
        local_path = row["local_path"]  # e.g. raw/OpenWorldCreative-v0/<seed>
        sha = row["sha256"]
        for name in ("data.npz", "rgb.mp4"):
            _copy(os.path.join(src, local_path, name), os.path.join(staging_dir, local_path, name))
        _copy(
            os.path.join(src, "voxel_classes", f"{sha}.npz"),
            os.path.join(staging_dir, "voxel_classes", f"{sha}.npz"),
        )

    # filtered metadata (preserve quoting; drop the index column)
    selected.to_csv(os.path.join(staging_dir, "metadata.csv"), index=False)

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(staging_dir)
        for f in fs
    ) / 1e6
    logger.info(f"Staged sample dataset at {staging_dir} ({size_mb:.1f} MB).")

    api = HfApi()
    if create:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=staging_dir, repo_id=repo_id, repo_type="dataset")
    logger.info(f"Uploaded sample dataset to https://huggingface.co/datasets/{repo_id}")


def upload_weights(
    namespace: str,
    voxel_encoder: str,
    voxel_decoder: str,
    pixel_vae: str,
    camera_model: str,
    voxel_denoiser_s: str,
    voxel_denoiser_xl: str,
    pixel_denoiser: str,
    voxel_class_embedder: str,
    repo_prefix: str = "persist-",
    staging_dir: str = "outputs/hf_weights",
    private: bool = False,
    create: bool = True,
) -> None:
    """Upload the checkpoints, one model repo per checkpoint.

    Each repo ``<namespace>/<repo_prefix><repo-suffix>`` gets a single ``model.safetensors`` — the
    canonical filename ``scripts.run_inference`` downloads. The voxel denoiser is the only
    variant-specific model: it ships as two repos, ``...-voxel-denoiser-s`` and
    ``...-voxel-denoiser-xl``; all other models are shared across the S/XL pipeline variants.

    Args:
        namespace: HF org/user (e.g. ``PERSIST-team``).
        voxel_denoiser_s / voxel_denoiser_xl: the S- and XL-variant voxel denoiser checkpoints.
        voxel_encoder, ...: Local path to each shared checkpoint's ``.safetensors``.
        repo_prefix: Prefix for the per-model repo names.
        staging_dir: Local folder assembled before upload (overwritten).
        private: Create repos private instead of public.
        create: Create each repo (``exist_ok=True``) before uploading.
    """
    from huggingface_hub import HfApi

    # repo-name suffix (dashed) -> local checkpoint path
    sources = {
        "voxel-encoder": voxel_encoder,
        "voxel-decoder": voxel_decoder,
        "pixel-vae": pixel_vae,
        "camera-model": camera_model,
        "voxel-denoiser-s": voxel_denoiser_s,
        "voxel-denoiser-xl": voxel_denoiser_xl,
        "pixel-denoiser": pixel_denoiser,
        "voxel-class-embedder": voxel_class_embedder,
    }
    missing = [s for s, p in sources.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Checkpoint file(s) not found for: {missing}")

    api = HfApi()
    for suffix, src_path in sources.items():
        repo_id = f"{namespace}/{repo_prefix}{suffix}"
        if create:
            api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        # Stage under the canonical filename, then upload that single file.
        staged = os.path.join(staging_dir, repo_id.replace("/", "__"), "model.safetensors")
        _copy(src_path, staged)
        api.upload_file(
            path_or_fileobj=staged,
            path_in_repo="model.safetensors",
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info(f"Uploaded {suffix} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "upload-eval-dataset": upload_eval_dataset,
            "upload-weights": upload_weights,
        }
    )
