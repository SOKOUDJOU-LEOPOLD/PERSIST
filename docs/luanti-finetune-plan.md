# PERSIST fine-tuning plan — Luanti external dataset (percool/luanti)

## Context

PERSIST has so far only been trained on this project's own generated data. The goal now is to
fine-tune PERSIST's full pipeline — voxel denoiser (**XL variant**, the largest/most powerful —
see note below), pixel denoiser, and camera model, all three, starting from the existing
pretrained checkpoints — on a new external dataset
(huggingface.co/datasets/percool/luanti), using a mix of 100 training episodes (50 from
`motworld-s1nav-10k`, 50 from `motworld-hudnav-20k`) and 15 validation episodes (8 from s1nav, 7
from hudnav), all drawn exclusively from episodes labeled `"split": "train"` in the dataset's own
split files. Training is tracked live via wandb, with a full pass over all 15 validation episodes
at each validation check (not just one random batch, which is what PERSIST's training code does
by default today).

Two deep research passes (one into the data-generation/loader pipeline, one into the training
scripts) plus a dedicated planning pass established that this is a genuinely multi-stage task:
the raw data format matches this repo's own pipeline almost exactly, pretrained checkpoints are
already cached locally, but several real gaps exist — no `dataset_params.json` ships with the
external dataset, no pretrained-checkpoint-init mechanism exists in PERSIST's training code (only
same-run resume), and no experiment tracking (wandb/tensorboard) exists in PERSIST's training
code at all. This document is the complete, ordered plan to close those gaps and run all three
fine-tunes.

## Phase 0 — Cheap gates before touching the big archives

**0.1 Checkpoint paths** (already confirmed cached locally, no download needed):

| Model | Path |
|---|---|
| voxel VAE encoder | `~/.cache/huggingface/hub/models--PERSIST-team--persist-voxel-encoder/snapshots/a9ff10b4a858d4e5e99aa784d7754b6f70c23396/model.safetensors` |
| pixel VAE | `~/.cache/huggingface/hub/models--PERSIST-team--persist-pixel-vae/snapshots/ac2c8ed622e3acaf071790eb4a021f0a367d82ae/model.safetensors` |
| voxel denoiser (**XL**) | `~/.cache/huggingface/hub/models--PERSIST-team--persist-voxel-denoiser-xl/snapshots/59e102280371f658ea2bee70f55315527d70b0f8/model.safetensors` |
| pixel denoiser | `~/.cache/huggingface/hub/models--PERSIST-team--persist-pixel-denoiser/snapshots/31bcbf1f5498ad5810003827c7a4156942bec8d1/model.safetensors` |
| voxel class embedder | `~/.cache/huggingface/hub/models--PERSIST-team--persist-voxel-class-embedder/snapshots/82444a59499b9a9f714226b897b544bc9364d758/model.safetensors` |
| camera model | `~/.cache/huggingface/hub/models--PERSIST-team--persist-camera-model/snapshots/490c951eaf727ee54fba4403bdf8e07c728b04c0/model.safetensors` |
| voxel decoder (frozen, for pixel training's conditioning) | `~/.cache/huggingface/hub/models--PERSIST-team--persist-voxel-decoder/snapshots/d62fe9b3b9a7def10e21e8fb9ab4dde6c2daf9d7/model.safetensors` |

**S vs XL, confirmed directly from code** (`pipelines/pipeline.py:27-47`, `scripts/run_inference.py:175-176`): the two pipeline variants differ in exactly one thing — the voxel denoiser's `patch_size` (2 for S, 1 for XL), which is what gives XL "8× more spatial tokens" (halving patch size ~triples tokens per axis in 3D). Everything else (`hidden_size=1024`, `depth=12`, `num_heads=16`, pixel VAE, pixel denoiser, camera model) is identical and shared between the two — **XL is not a bigger backbone, it's the same backbone at finer spatial resolution**, trained as a separate checkpoint. `noise_cond=0.0` is XL's other listed difference but is a `SamplerArgs` field — purely an inference-time sampling setting, irrelevant to training. **Practical consequence, confirmed against `train_scripts/train_diffuser_voxel.py --help`'s actual CLI**: the voxel-denoiser fine-tuning launch (Phase 7) must pass `--denoiser.patch-size 1` explicitly — the default is 2 (S-shaped), and leaving it at default while loading XL's pretrained weights would crash `load_state_dict` on a shape mismatch.

All three pretrained checkpoints (voxel encoder/decoder, camera model) are hard-shaped to a 2138-class voxel vocabulary (`ResNetEncoderArgs.in_channels=2138`, `CameraTransformerArgs.voxel_vocab_size=2138`) — this is why reusing `dataset_toolkits/default_mt_voxel_classdict.json` (confirmed 2138 entries) is a hard requirement, not just a convenience; a freshly-generated classdict would produce incompatible shapes.

**0.2 Download only the two `.splits.json` files** (1.2MB/2.5MB — not the archives):
```bash
uv run python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["motworld-s1nav-10k.splits.json", "motworld-hudnav-20k.splits.json"]:
    print(f, "->", hf_hub_download(repo_id="percool/luanti", filename=f, repo_type="dataset"))
PY
```

**0.3 Empirical classdict-coverage check** on 2-3 real sample episodes per source (stream-extract only those, into a throwaway directory, treated as untrusted data):
```bash
zstd -dc motworld-s1nav-10k.tar.zst | tar -xf - --wildcards 'OpenWorldCreative-v0/<seed>/*' -C /tmp/sample_check/
```
Load each sample's `obs_voxel_mt`, crop to 48³ (matching `process_mt_data.py`'s own crop), collect unique `(node_id, node_param)` pairs, and check what fraction is covered by `dataset_toolkits/default_mt_voxel_classdict.json`. **Decision gate**: high coverage (>95%, missing pairs look rare/decorative) → proceed with `--default-classdict` reuse. Poor/systemic coverage → stop and reassess before committing to the full 115-episode ingestion (a classdict mismatch would silently break pretrained-VAE reuse, since `process_mt_data.py` invalidates any episode containing even one unknown voxel pair).

## Phase 1 — Episode selection (new script)

New script `dataset_toolkits/select_luanti_episodes.py`:
1. Load the two `.splits.json` files (Phase 0.2), filter each to `split == "train"` only.
2. Deterministic `random.Random(<fixed seed>)`, logged for reproducibility.
3. Sample 50 s1nav + 50 hudnav for training; from the remaining `"train"`-labeled pool, sample **8 s1nav + 7 hudnav** for validation.
4. Oversample a few spares per pool (e.g. +5 train, +2 val) as a buffer against Phase 0.3-style classdict rejections during actual ingestion, without needing to re-sample and perturb the manifest.
5. Write a manifest (`dataset_toolkits/manifests/luanti_finetune_manifest.json`) recording per-episode `{sha256, seed, source_env, split_role, archive_name}` plus the sampling seed and split-file provenance — per the standing paper-citation logging preference. Also emit flat seed-list files (`s1nav_seeds.txt`, `hudnav_seeds.txt`) for Phase 2's extraction command.

## Phase 2 — Selective streaming extraction (not full archive unpacking)

One batched `tar --wildcards` pass per archive (so `zstd` decompresses once per source, not once per episode), landing exactly 115 episode directories, then moved into `datasets/luanti_finetune/raw/<env_id>/<seed>/...` — the layout `build_dataset.sh` expects. Verify count: `find datasets/luanti_finetune/raw -mindepth 2 -maxdepth 2 -type d | wc -l` should print `115`.

## Phase 3 — `dataset_params.json`

Confirmed `generate_raw_data.py --init` does **not** require a live Craftium/Minetest env (the `--init` path exits before any `gym.make` call) — so the simplest, most consistent approach is running it directly rather than hand-authoring the JSON:
```bash
uv run python dataset_toolkits/generate_raw_data.py \
  --dataset_dir datasets --dataset_name luanti_finetune \
  --voxel_obs_rx 24 --voxel_obs_ry 24 --voxel_obs_rz 24 \
  --num_levels 1 --init
```
This writes a correct `dataset_params.json` (radius-24 → raw 49³, cropped target 48³, matching the external dataset's documented shape) without needing `OpenWorldCreativeHud-v0` to be a registered env (which it currently isn't — confirmed, but irrelevant here since this only matters for *live generation*, not for consuming pre-recorded episodes).

## Phase 4 — `build_dataset.sh`'s steps, run individually (not the monolithic script)

The stock script's final step (`--set_validation_samples vae`) auto-assigns a random 5% tail-slice as validation — incompatible with needing an exact, specific 15-episode set. Run the script's other steps manually, substituting a new custom step for the split assignment:
```bash
source .venv/bin/activate
DS=datasets/luanti_finetune

bash dataset_toolkits/check_missing_data.sh "$DS" -d
python dataset_toolkits/build_metadata.py --output_dir "$DS" --from_file
cp dataset_toolkits/default_mt_voxel_classdict.json "$DS/mt_voxel_classdict.json"
python dataset_toolkits/process_mt_data.py --output_dir "$DS" --check_conditions_only --overwrite
python dataset_toolkits/build_metadata.py --output_dir "$DS"
python dataset_toolkits/process_mt_data.py --output_dir "$DS"
python dataset_toolkits/build_metadata.py --output_dir "$DS" --from_file   # no --set_validation_samples
```
Then a new script `dataset_toolkits/assign_validation_split.py` (`--dataset_dir`, `--val_sha256_file`): loads `metadata.csv`, asserts every val sha256 actually survived preprocessing (`voxelized==True`, `level_quality_score>0`) — failing loudly rather than silently substituting if one didn't — sets `validation_set=True` for exactly those 15, `False` for the rest, writes it back.

**Verify**: `validation_set` column sums to exactly `15 True / 100 False`.

## Phase 5 — `build_latents.sh` and `compute_latent_stats.py`

No code changes needed — both already work as-is once `metadata.csv` is correct:
```bash
bash dataset_toolkits/build_latents.sh datasets/luanti_finetune \
  --voxel-vae-path <voxel encoder path from 0.1> \
  --pixel-vae-path <pixel VAE path from 0.1>

python dataset_toolkits/compute_latent_stats.py --dataset_path datasets/luanti_finetune --process_pixel --process_voxel
```
(`compute_latent_stats.py` already automatically restricts to `validation_set == False` rows — confirmed no data leakage regardless of this custom split.)

## Phase 6 — Training-script code changes (shared, in `trainers/`, not duplicated per script)

All three target scripts (`train_diffuser_voxel.py`, `train_diffuser_pixel.py`, `train_transformer_camera.py`) share an identical `main()` skeleton and subclass the same `BaseTrainer` — so all three changes land once in `trainers/__init__.py`.

**6.1 Pretrained-checkpoint-init** (new — distinct from the existing resume mechanism, which always restores the step counter): new `BaseTrainingArgs` fields `pretrained_checkpoint: str | None = None` and `pretrained_checkpoint_model_key: str = "denoiser"`; new `BaseTrainer.load_pretrained_checkpoint()` method doing a raw `safetensors.load_file` + `load_state_dict(strict=False)` into `self.models[key]` — the same idiom already proven twice in this codebase (`pipelines/pipeline.py`'s `from_pretrained`, `train_diffuser_pixel.py`'s existing `load_voxel_decoder()`). Called in each script's `main()` right after `prepare_models()` and before `prepare_trainable_parameters()`/`prepare_accelerate()`, so weights load into the unwrapped model before Accelerate wraps anything, and the step counter stays at 0.

**Confirmed model-key mapping**: `train_diffuser_voxel.py` → `self.models["denoiser"]` (its only key). `train_transformer_camera.py` → `self.models["camera"]`. `train_diffuser_pixel.py` → `self.models["denoiser"]` **and** `self.models["voxel_class_embedder"]` (confirmed via direct code read: `prepare_models()` builds `nn.ModuleDict({"denoiser": ..., "voxel_class_embedder": nn.Embedding(...)})`, and the embedder currently has no loading path at all — always randomly initialized). Per the confirmed decision, extend `load_pretrained_checkpoint()` to accept loading into a second model key (or call it twice) so the pixel-denoiser run is a genuine full fine-tune of both trainable components, not a fine-tuned denoiser paired with a from-scratch embedder.

**6.2 wandb integration** (new — confirmed zero experiment tracking exists in `trainers/`/`train_scripts/` today, though `wandb>=0.25.0` is already a declared dependency). New `BaseTrainingArgs` fields: `use_wandb: bool = False` (default off, so other non-finetune configs are unaffected), `wandb_project`, `wandb_entity: str | None = "leopoldgatsing-texas-a-m-university"`, `wandb_run_name`. In `BaseTrainer.train()`: `wandb.init(...)` gated to `is_main_process` near the top, `wandb.log(log_values, step=self.step)` at the existing `i_log` cadence block, `wandb.finish()` at the end — same gating pattern already proven twice this session for the Oasis/WorldMem baselines (`baselines/finetune_oasis.py`).

**6.3 Full-pass validation** (new, non-breaking opt-in). New field `full_pass_validation: bool = False` (default preserves today's single-random-batch behavior for any run not part of this task). When set, the validation block in `train()` iterates the *entire* `self.val_dataloader` (confirmed already correctly constructed with `split="val"` in each script's `prepare_dataset()`, which filters `metadata.csv`'s `validation_set == True` — exactly Phase 4's 15 rows) and averages loss/PSNR terms across the full pass, instead of drawing one batch via `next()` on an infinite iterator. No new dataset-loading code needed — purely a change to how `train()` consumes the existing val dataloader.

## Phase 7 — Launch the three fine-tunes, sequentially

Per the confirmed decision: one at a time (no hard dependency between the three — they all train independently against the same pre-encoded ground-truth dataset, not each other's live outputs — sequential is chosen for GPU/disk-sharing discipline and to sanity-check each stage before committing to the next, matching this session's established pattern).

**Order**: voxel denoiser → camera model → pixel denoiser (natural pipeline order for qualitative sanity-checking after each stage, not a hard requirement).

```bash
# 1. Voxel denoiser (XL variant -- note --denoiser.patch-size 1, required, see Phase 0.1)
uv run accelerate launch --mixed_precision bf16 --num_processes <N_GPU> \
  train_scripts/train_diffuser_voxel.py \
  --run-name luanti-finetune-voxel-denoiser-xl --output-dir outputs/runs/voxel_diffuser \
  --dataset-path datasets/luanti_finetune \
  --pretrained-checkpoint <voxel denoiser XL path> --pretrained-checkpoint-model-key denoiser \
  --denoiser.patch-size 1 \
  --use-wandb --wandb-project persist-luanti-finetune --wandb-run-name voxel-denoiser-xl \
  --full-pass-validation

# 2. Camera model
uv run accelerate launch --mixed_precision bf16 --num_processes <N_GPU> \
  train_scripts/train_transformer_camera.py \
  --run-name luanti-finetune-camera-model --output-dir outputs/runs/camera_transformer \
  --dataset-path datasets/luanti_finetune \
  --pretrained-checkpoint <camera model path> --pretrained-checkpoint-model-key camera \
  --use-wandb --wandb-project persist-luanti-finetune --wandb-run-name camera-model \
  --full-pass-validation

# 3. Pixel denoiser (loads BOTH denoiser + voxel_class_embedder pretrained weights, per confirmed decision)
uv run accelerate launch --mixed_precision bf16 --num_processes <N_GPU> \
  train_scripts/train_diffuser_pixel.py \
  --run-name luanti-finetune-pixel-denoiser --output-dir outputs/runs/pixel_diffuser \
  --dataset-path datasets/luanti_finetune \
  --voxel-decoder-checkpoint <voxel decoder path> \
  --pretrained-checkpoint <pixel denoiser path> --pretrained-checkpoint-model-key denoiser \
  --voxel-class-embedder-checkpoint <voxel class embedder path> \
  --use-wandb --wandb-project persist-luanti-finetune --wandb-run-name pixel-denoiser \
  --full-pass-validation
```
All three launches use the `diffusion-forcing` subcommand (e.g. `... train_diffuser_voxel.py [flags] diffusion:diffusion-forcing`) — confirmed via `trainers/flow_euler.py:24-29`: `DiffusionForcingArgs` (`noise_cond=0.015`, `sigma_min=1e-5`, `noise_abs_max=20.0`) is this project's actual training-time methodology (matching "Diffusion Forcing", the framework this whole codebase is built around), versus the bare, parameter-free `DiffusionTeacherForcingArgs` alternative.

## Verification checklist

- Phase 0.3: classdict coverage high, missing pairs rare/decorative.
- Phase 2/3: exactly 115 raw episode directories.
- Phase 4 (before split assignment): `voxelized`/`level_quality_score>0` counts close to 115 (shortfall → trace back to Phase 1's spares).
- Phase 4 (after split assignment): `validation_set` sums to exactly `15 True / 100 False`.
- Phase 5: `voxel_latents/`/`pixel_latents/` populated for all 115 sha256s; `metadata.csv`'s `*_latent_generated` columns all `True`.
- Phase 7: wandb shows live-updating loss curves per run; first validation trigger completes without error and its wall-clock time scales with ~15 episodes' worth of batches (not 1), confirming the full-pass code path is actually being exercised.

## Critical files

- `dataset_toolkits/build_dataset.sh`, `build_metadata.py`, `process_mt_data.py`, `generate_raw_data.py`, `default_mt_voxel_classdict.json`
- `trainers/__init__.py` (all three code changes land here), `trainers/flow_euler.py`
- `train_scripts/train_diffuser_voxel.py`, `train_diffuser_pixel.py`, `train_transformer_camera.py`
- `data_loaders/minetest_latent_camera_action_dataset.py`
- `baselines/finetune_oasis.py` (wandb pattern reference), `pipelines/pipeline.py` (pretrained-load idiom reference)
