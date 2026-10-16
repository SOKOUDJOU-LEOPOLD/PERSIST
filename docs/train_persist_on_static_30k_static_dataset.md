# PERSIST: retrain voxel denoiser (XL) + camera model + pixel denoiser on the full Luanti dataset

## Context

We already fine-tuned all three trainable PERSIST components (voxel denoiser XL, camera model,
pixel denoiser — the two VAEs stay frozen/pretrained) on a **115-episode pilot** built from the
HuggingFace dataset `percool/luanti` (`motworld-s1nav-10k` + `motworld-hudnav-20k`, mixed). That
pilot validated the entire pipeline end-to-end: data selection → preprocessing → pretrained-weight
initialization → training with wandb → full-pass validation → inference. All three runs completed
5,000 steps cleanly, checkpoints were saved, and we ran working inference rollouts (16-frame and
600-frame) on held-out training episodes using the fine-tuned weights.

**Now we want to scale this up**: retrain the same three components on as much of the *full*
`motworld-s1nav-10k` + `motworld-hudnav-20k` "train"-split pool as possible (9,463 s1nav + 19,545
hudnav = 29,008 episodes total), with a much larger held-out validation set (200 episodes: 100
s1nav + 100 hudnav). The original server this was being planned on turned out not to have enough
disk for this at full scale — this document is written to be handed to a different
machine/agent that may have more room, so it's self-contained rather than assuming shared context.

**If you're running this on a different server, do your own disk math first** (see "The
caching-vs-on-the-fly decision" below) — the numbers in this doc are specific to the original
H100 machine's measured rates and may not transfer directly, especially GPU timing numbers.

## What's already proven working (the pilot)

These are real, already-implemented, already-tested code changes in this PERSIST checkout. If
you're working from a *different* copy of the repo, port these first — don't re-derive them from
scratch:

1. **Pretrained-checkpoint-init mechanism** (new code, not stock PERSIST). `trainers/__init__.py`:
   `BaseTrainingArgs` gained fields `pretrained_checkpoint`, `pretrained_checkpoint_model_key`,
   `pretrained_checkpoint_extra`, `pretrained_checkpoint_extra_model_key`, `use_wandb`,
   `wandb_project`, `wandb_entity`, `wandb_run_name`, `full_pass_validation`. `BaseTrainer` gained
   `load_pretrained_checkpoint()` (loads a `safetensors` file into the raw unwrapped model via
   `load_state_dict(strict=False)`, strips an `_orig_mod.` prefix if present, supports a second
   model key for e.g. the pixel denoiser's `voxel_class_embedder`). This is called in each
   `train_scripts/*.py`'s `main()` right after `trainer.prepare_models()` and before
   `trainer.prepare_trainable_parameters()` — critically, *before* Accelerate wraps the model, and
   it does **not** touch the step counter (unlike the stock `--resume`/`load_state`, which is for
   resuming an interrupted run of the *same* config, not initializing a new run from different
   pretrained weights).
2. **wandb + full-pass validation** (new code). `train()` in `trainers/__init__.py` now calls
   `wandb.init(...)` (gated on `use_wandb and is_main_process`) and logs via `wandb.log(...)`.
   Validation, when `full_pass_validation=True`, iterates the *entire* `val_dataloader` and
   averages metrics, instead of the stock behavior (one random batch via `next()` on an infinite
   iterator).
3. **Known required CLI overrides, not defaults** — confirmed by shape-mismatch crashes during the
   pilot, not guesses:
   - Voxel denoiser **XL** variant: `--denoiser.patch-size 1` (script default is `2`, which is the
     **S** variant's shape — S and XL share every other config; patch_size is the *only*
     architectural difference, confirmed in `pipelines/pipeline.py:27-47`).
   - Camera model: `--model.cam-rotation-output-rep delta_angle --model.cam-fov-output-rep delta`
     (script defaults are `6d`/`absolute`, which don't match the pretrained checkpoint's actual
     trained representation — confirmed against `pipelines/persist_S_pipeline/model_configs.py`'s
     `CameraTransformerArgs`, the known-good inference config).
   - Camera model also asserts `i_log % i_val == 0` (`train_transformer_camera.py:126`) — a
     constraint unique to this script; we used `--i-log 100 --i-val 100`.
4. **`cropped_voxel` must be precomputed** before camera-model training: run
   `process_mt_data.py --compute_cropped_voxels` once per dataset (cheap additive pass, ~30s for
   115 episodes, no re-extraction needed) — without it, camera-model training hits an infinite
   "resampling another instance" loop that eventually crashes with a recursion error.
5. **`ninja` is required** for the pixel denoiser (JIT-compiles `nvdiffrast`'s CUDA extension for
   voxel rasterization): `uv add ninja`, and the launch command needs an **activated venv**
   (`source .venv/bin/activate`), not just `PYTHONPATH=.` — `ninja` has to be on `PATH`.
6. **`PYTHONPATH=.`** (repo root) is required for every `dataset_toolkits/*.py` and
   `train_scripts/*.py` invocation, including inside background/subprocess workers launched by
   `accelerate launch` — not handled automatically by `uv run` or a bare `.venv/bin/python`.
7. **Checkpoints use EMA, but don't trust it blindly at low step counts.** `ema_decay=0.9999`
   means after only 5,000 steps, the EMA weights are still ~60% anchored to the pretrained
   initialization (`decay^steps ≈ e^{-steps·(1-decay)} ≈ e^{-0.5} ≈ 0.61`). We used the raw
   `model.safetensors` (not `ema.safetensors`) for inference on the pilot for this reason. At a
   much larger step count (this run), re-evaluate — EMA becomes meaningful over longer schedules.
8. **Inference eval loader had a real bug, now fixed.** `data_loaders/minetest_latent_camera_action_dataset.py`'s
   `MinetestLatentCameraActionEval.filter_metadata` used to *always* restrict to `validation_set ==
   True`, even when specific episode hashes were requested via `--eval-instances` — meaning
   training-split episodes could never be loaded for inference at all, contradicting the script's
   own documented contract. Fixed: the validation-only restriction now only applies when no
   explicit instances are requested (the default case).

## Pretrained checkpoint sources (HF, `PERSIST-team` org — universal, not server-specific)

| Component | Trainable this round? | HF repo |
|---|---|---|
| Voxel encoder | No (frozen) | `PERSIST-team/persist-voxel-encoder` |
| Voxel decoder | No (frozen) | `PERSIST-team/persist-voxel-decoder` |
| Pixel VAE | No (frozen) | `PERSIST-team/persist-pixel-vae` |
| Voxel denoiser (XL) | **Yes** | `PERSIST-team/persist-voxel-denoiser-xl` |
| Camera model | **Yes** | `PERSIST-team/persist-camera-model` |
| Pixel denoiser | **Yes** | `PERSIST-team/persist-pixel-denoiser` (+ `model_1.safetensors` = voxel class embedder) |

Each repo holds a single `model.safetensors`. Resolve via `huggingface_hub.hf_hub_download` or let
`scripts/run_inference.py`'s existing `--checkpoints-namespace PERSIST-team` resolution logic do it
(see `resolve_checkpoints()` there for the exact naming convention, including the `-xl` suffix
rule for the voxel denoiser specifically).

## Data specification

- Source: HF dataset `percool/luanti`, archives `motworld-s1nav-10k.tar.zst` (env
  `OpenWorldCreative-v0`) and `motworld-hudnav-20k.tar.zst` (env `OpenWorldCreativeHud-v0`, HUD
  rendered into frame).
- **Only** episodes labeled `"split": "train"` in each archive's own `*.splits.json` — never touch
  `"eval"`. Confirmed pool sizes (train-split only): **9,463 s1nav / 19,545 hudnav** (29,008
  total).
- **Validation set: 200 episodes, 100 s1nav + 100 hudnav**, held out from training, drawn only
  from the "train"-split pool (same rule as training — never touch the dataset's own "eval"
  split).
- **Training set: as many of the remaining train-split episodes as disk allows**, mixed
  proportionally across both sources (not necessarily 50/50 — the two sources aren't the same
  size: roughly 1 s1nav : 2 hudnav by pool size, matching their natural ratio, unless you have a
  specific reason to over/under-sample one source).
- Reuse pattern from the pilot: `dataset_toolkits/select_luanti_episodes.py` already implements
  deterministic sampling from the `.splits.json` files with a seed, writing a manifest +
  `_seeds.txt` files — generalize it to the new target counts rather than rewriting from scratch.
  Oversample a small spare buffer per source (we used +5 train/+2 val in the pilot) as a buffer
  against classdict-coverage rejections (see below) — at this much larger scale, the buffer should
  probably be a percentage of the target (e.g. 2-3%) rather than a small fixed number, since
  rejection counts scale with episode count, not stay constant.

## Critical constraint: voxel vocabulary compatibility (do not skip this)

The frozen voxel VAE (encoder + decoder) and the camera model are **hard-shaped** to a
**2,138-class voxel vocabulary** (`ResNetEncoderArgs.in_channels=2138`,
`CameraTransformerArgs.voxel_vocab_size=2138`). This is baked into the pretrained weights — it
cannot be changed without retraining those frozen components from scratch, which is explicitly out
of scope here.

**The pilot's approach, which must be repeated exactly**: seed the new dataset's
`mt_voxel_classdict.json` with a **copy of** `dataset_toolkits/default_mt_voxel_classdict.json`
(the pretrained vocabulary) *before* running `build_metadata.py`'s classdict-merge step
(`cp dataset_toolkits/default_mt_voxel_classdict.json "$DS/mt_voxel_classdict.json"`, then let
`process_mt_data.py` run against that seeded file). `process_mt_data.py` then **rejects (not
silently corrupts) any episode containing even one voxel `(content, param2)` pair not already in
the seeded 2,138-class vocabulary** — it does not grow the vocabulary. This is a safety behavior
that must not be worked around.

**Implication for this larger run**: at 115 episodes, coverage was checked on only 2-3 sample
episodes per source and found to be >95% (rare/decorative pairs missing). At ~100-250x more
episodes, expect a **higher absolute rejection count** even if the rejection *rate* stays similar
— budget for this with a real spare buffer (see above), and re-run the pilot's "Phase 0.3" coverage
check (sample a larger batch, e.g. 50-100 episodes per source, check coverage against the seeded
2,138-class dict) before committing to full-scale extraction, rather than trusting the small
pilot's coverage number to hold at this scale.

## The caching-vs-on-the-fly decision (disk vs. compute time tradeoff)

The pilot's dataset cached fully-precomputed VAE latents to disk (`voxel_latents/`,
`pixel_latents/` — continuous, uncompressed float tensors, one per frame). This is PERSIST's stock
approach (`data_loaders/minetest_latent_camera_action_dataset.py`, all three `train_scripts/*.py`
assume cached latents exist on disk and read them directly). Measured on the 115-episode pilot:

- **Per-episode footprint with cached latents: ~155 MB** (raw ~11MB + voxel_classes ~5.5MB +
  voxel_latents ~96MB + pixel_latents ~42MB). At 29,008 episodes, that's **~4.5 TB**.
- **Per-episode footprint without cached latents (raw + voxel_classes + cropped_voxel only): ~16
  MB.** At 29,008 episodes, that's **~462 GB**.

**Decide this based on your own disk, not ours.** Run `df -h` on every writable mount you actually
have *write permission* on (don't assume — we wrongly assumed the root partition was usable on the
original machine and it wasn't; verify with `stat -c "%a %U %G" <path>` or a real write test, not
just `df -h`'s raw numbers) and compare against the per-episode rates above at your target episode
count.

**If you have comfortably more disk than ~4.5TB available**: just use the stock cached-latent
pipeline exactly as the pilot did — no code changes needed, already proven working. Skip the rest
of this section.

**If you don't** (our situation): switch to on-the-fly VAE encoding — run the frozen voxel/pixel
VAE encoders live, inside the training loop, on raw `voxel_classes`/images, instead of reading
precomputed latents. **This is not yet implemented** — it requires real code changes:

- `data_loaders/minetest_latent_camera_action_dataset.py` and all three `train_scripts/*.py`
  currently assume "read a precomputed latent tensor from a cached `.safetensors` file." They'd
  need to instead load raw `voxel_classes` (class-ID grids) and raw pixel frames, and run them
  through the frozen VAE encoders (loaded onto the GPU, `requires_grad=False`, wrapped in
  `torch.no_grad()`) as part of each training step, to produce the same latent tensors on the fly.
  Both encoders stay **completely frozen** — forward pass only, no backward pass, never registered
  with the optimizer, same as how the pixel-denoiser trainer already treats its frozen voxel
  *decoder* today (`train_diffuser_pixel.py`'s `load_voxel_decoder()`/`decode_voxel_latents_batch`
  is the existing pattern to mirror).
- Latent **normalization** (`normalize_latents=True`, via `latent_stats.npz` mean/std) still needs
  to apply to the freshly-computed latents for consistency with how the pretrained models were
  trained — this doesn't require caching all episodes, just computing (or reusing) representative
  mean/std stats once.
- Extra GPU memory: the frozen VAEs need to sit in GPU memory alongside the trainable model during
  training. Modest relative to the trainable models (see parameter counts below), but not zero.

**Measured compute cost of going on-the-fly** (on one H100 NVL, `batch_size=2`, matching the
pilot's actual per-step frame counts — re-measure on your own GPU if different hardware):

| | Frames/step | VAE forward add | Baseline step (cached) | New total/step | Overhead |
|---|---|---|---|---|---|
| Voxel denoiser (context_window=8) | 16 | +158.6ms (104.5ms voxel-enc + 54.1ms pixel-enc) | 1.17s | 1.329s | **+13.6%** |
| Pixel denoiser (context_window=16) | 32 | +321.1ms (212.6ms voxel-enc + 108.5ms pixel-enc) | 1.055s | 1.376s | **+30.4%** |

(Camera model is unaffected either way — it never uses either VAE; it trains on `camera`,
`action`, and `cropped_voxel` only, `no_latents=True`.)

For reference, trainable-model parameter counts (relevant to how VAE overhead compares):
voxel denoiser XL 686M, pixel denoiser 460M, camera model 234M, vs. frozen voxel encoder ~62M,
voxel decoder ~76M, pixel VAE ~228M.

**Why on-the-fly is very likely worth it even with that overhead**: the eliminated cost isn't "zero
vs. a little slower" — it's "a one-time preprocessing pass that scales with dataset size, vs. a
modest per-step training slowdown that scales with step count." Measured on the pilot: building
`voxel_latents/` + `pixel_latents/` for 115 episodes took **~23.8 minutes** (~12.4s/episode). At
29,008 episodes, that's **~4.2 days** of preprocessing *before training can even start* — vs. the
per-step overhead above, which over a 5,000-step run only adds ~13-27 minutes total, or ~1-2 hours
even at 20,000 steps. Do this estimate again on your own hardware (encode-pass throughput and
training step time both depend on GPU), but expect the same shape of result: on-the-fly wins
decisively once the dataset gets large, on both disk *and* total wall-clock.

## Checkpoint retention (no policy exists today — decide based on your disk)

`trainers/__init__.py`'s `save_state()` keeps every checkpoint saved at the `i_save` interval
forever (confirmed: no cleanup/rolling-deletion logic anywhere in the codebase). Each checkpoint
folder holds `model.safetensors` + `ema.safetensors` + `optimizer.bin` (+ RNG state). Measured
sizes per full checkpoint round (all 3 components, one save each):

| Component | model+ema | +optimizer (full round) |
|---|---|---|
| Voxel denoiser XL | ~5.49GB | ~10.98GB |
| Pixel denoiser | ~3.68GB | ~7.36GB |
| Camera model | ~1.87GB | ~3.73GB |
| **Total/round** | **~11.0GB** | **~22.1GB** |

Optimizer state is only needed to *resume* training, not for inference or comparison. A simple,
cheap win: only keep optimizer state in the **latest** checkpoint (delete it from older ones once
a newer checkpoint lands), keep `model.safetensors`+`ema.safetensors` for every saved checkpoint —
roughly halves checkpoint disk cost with no loss of inference/comparison capability, at the cost of
only being able to resume from the single latest checkpoint. Not yet implemented — would require a
small change to `save_state()` (or a cleanup step after each save).

Pick an `i_save` frequency based on: (your available checkpoint disk budget) ÷ (per-round cost at
your chosen retention policy) ≈ how many checkpoints you can afford across the full run.

## Open decisions (not yet settled — whoever picks this up needs to choose)

1. **Final training-set episode count** — capped by whatever disk is actually available on the
   target machine. Full pool is 29,008 (minus 200 held out for validation = 28,808 training). If
   disk doesn't stretch that far, scale down proportionally across both sources.
2. **Total training steps per component** — the pilot used 5,000 steps on 115 episodes. This
   dataset will be far larger; decide whether to keep 5,000 (far less per-episode repetition, far
   more distinct data) or scale up (closer to the pilot's per-episode exposure, proportionally
   longer runtime).
3. **Checkpoint frequency (`i_save`) and retention policy** — depends on #1 (training length) and
   available checkpoint disk budget; see table above.
4. **Caching vs. on-the-fly** — depends entirely on the target machine's actual writable disk (see
   decision section above). If on-the-fly is chosen, the code changes described there are new work,
   not yet built anywhere.

## Verification plan (once implemented)

Mirror what the pilot already validated successfully:
1. Confirm classdict coverage on a larger sample (50-100 episodes/source) before full extraction.
2. Launch all three fine-tunes (pretrained-init, not from scratch) with wandb tracking and
   full-pass validation over the full 200-episode val set.
3. Check learning curves make sense (watch for the camera-model-style overfitting signature we saw
   in the pilot: train loss crashing to near-zero while val stays flat/noisy — if that recurs at
   this larger scale, it's worth investigating rather than assuming it's expected).
4. Run `scripts/run_inference.py` (XL variant) on a couple of held-out training episodes, same
   pattern as the pilot, to sanity-check the fine-tuned weights produce coherent rollouts before
   calling the run done.
