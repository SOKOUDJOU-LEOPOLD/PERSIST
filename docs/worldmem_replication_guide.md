# WorldMem/Craftium fine-tuning: replication guide for a fresh server

Written for: an agent setting up this same WorldMem fine-tuning pipeline on a different
server/PERSIST checkout, from a fresh WorldMem clone. This document is the complete, precise
account of every modification made to support fine-tuning WorldMem on Craftium data in this repo.

## The one fact that matters most: WorldMem's own source was never touched

`baselines/WorldMem/` is its own nested git checkout of `https://github.com/xizaoqu/WorldMem.git`
(gitignored from this outer repo, exactly like `baselines/open-oasis/` is for the Oasis baseline).
Verified directly on the original server:

```
$ git -C baselines/WorldMem status --porcelain --ignored
(completely empty -- no modified, untracked, or ignored files anywhere)
$ git -C baselines/WorldMem log --oneline -3
c1cc91b Revise citation for WorldMem paper
e7987a2 fix psnr calculation
63a4fd0 update
$ git -C baselines/WorldMem diff origin/main --stat
(empty)
```

**Replication step 1: `git clone https://github.com/xizaoqu/WorldMem.git baselines/WorldMem` at
or near commit `c1cc91b`, and change nothing inside it.** Every piece of Craftium-specific
behavior described below lives in `baselines/*.py` files that sit *beside* the clone, which
`sys.path.insert(0, ".../WorldMem")` and `from algorithms.worldmem import WorldMemMinecraft` —
the same external-wrapping pattern `baselines/finetune_oasis.py` uses for `baselines/open-oasis/`.
No WorldMem yaml config was edited either — `configurations/huggingface.yaml` is loaded stock via
`OmegaConf.load` and overridden at runtime with two field mutations in Python (see below), never
routed through WorldMem's own Hydra/Lightning `main.py` entrypoint.

## Environment: WorldMem needs its own separate venv

WorldMem's own `requirements.txt` pins `torch~=2.4.0`, `lightning~=2.1.2`, `hydra-core~=1.3.2`,
`omegaconf~=2.3.0`, `rotary_embedding_torch`, etc. — this is **incompatible** with this repo's
main Oasis-pipeline venv, which pins `torch==2.7.1`. Confirmed directly: a `pip install --dry-run`
of `omegaconf lightning rotary_embedding_torch` into the main venv wants to pull in `torch-2.14.0`
and `pytorch-lightning-2.6.6`, overriding the pinned torch — so do not install WorldMem's deps
into the same venv the Oasis pipeline uses.

This repo's `pyproject.toml` now has an opt-in `worldmem` dependency group (never auto-installed,
`default-groups = []`) declaring `omegaconf~=2.3.0`, `lightning~=2.1.2`, `rotary_embedding_torch` —
useful as a reference for exact pinned versions, but build a **separate** venv for WorldMem work,
either from WorldMem's own `requirements.txt` directly, or `uv sync --group worldmem` into a venv
that does *not* also have the `cu` group's `torch==2.7.1` installed. `hydra-core` is listed in
WorldMem's own requirements but is **not actually needed** here — every script below bypasses
Hydra entirely (see above).

## Files to copy as siblings of `baselines/WorldMem/` (not inside it)

- `baselines/finetune_worldmem.py` — the fine-tuning script.
- `baselines/finetune_worldmem_data.py` — `CraftiumWorldMemDataset`.
- `baselines/finetune_worldmem_generate.py` — shared generation shell + the critical bug fix (see below).
- `baselines/craftium_action_features.py` — **shared with the Oasis pipeline**; the WorldMem half is detailed below.
- `baselines/eval_worldmem_heldout_loss.py`, `eval_worldmem_finetuned_fvd.py`, `eval_worldmem_pose_causality.py`, `eval_worldmem_pose_causality_cleancam.py`, `eval_worldmem_zeroshot.py`, `zeroshot_minecraft_worldmem.py`, `make_comparison_grid_worldmem.py`, `worldmem_validate_example.py`.
- `docs/action_space_comparison.md`, `docs/action_space_empirical_report.md` — the two reference docs the mapping code cites.
- `outputs/finetune_split.json` — train/test level-ID split every script above depends on (regenerated this session via `scripts/make_finetune_split.py`'s default args — fixed seed=0, reproduces the canonical 236 train / 20 test split exactly; also copy `scripts/make_finetune_split.py` itself if the new server needs to regenerate it independently).

## 1. `baselines/finetune_worldmem.py` — the fine-tuning recipe

### Model surgery (analogous to Oasis's `GatedActionEmbed` swap)

`load_worldmem_for_finetuning` constructs `WorldMemMinecraft` directly from the stock config with
two fields overridden **before** construction:

```python
cfg = OmegaConf.load(config_path)          # stock configurations/huggingface.yaml, unedited
cfg.action_cond_dim = action_dim           # stock value: 25; overridden to 23 or 25 (see layouts below)
cfg.require_pose_prediction = False        # stock value: true
worldmem = WorldMemMinecraft(cfg)

action_layer = worldmem.diffusion_model.model.external_cond
if zero_init_action_layer:
    torch.nn.init.zeros_(action_layer.weight)
    torch.nn.init.zeros_(action_layer.bias)
```

The pretrained checkpoint (downloaded via `hf_hub_download`, repo IDs from stock
`huggingface.yaml`) is loaded `strict=False`, explicitly dropping `model.external_cond.*` keys as
the one expected mismatch:

```python
mismatched = {k for k in state_dict if k.startswith("model.external_cond.")}
filtered = {k: v for k, v in state_dict.items() if k not in mismatched}
missing, unexpected = worldmem.diffusion_model.load_state_dict(filtered, strict=False)
expected_missing = {k for k in missing if "external_cond" in k} | {k for k in missing if k.endswith("rotary_emb.freqs")}
assert not (set(missing) - expected_missing)
assert not unexpected
```

Then `GatedActionEmbed` (a verbatim port of Oasis's own class, same fields/forward) is substituted
in by plain attribute assignment — `worldmem.diffusion_model.model.external_cond = gated` — no
WorldMem source touched:

```python
class GatedActionEmbed(torch.nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(action_dim, hidden_size)
        self.gate = torch.nn.Parameter(torch.ones(1))
    def forward(self, action):
        return self.linear(action) * self.gate
```

**The zero-gradient bug and its fix** (same bug, same fix, in both the Oasis and WorldMem
versions of this class): `gate` originally started at `0`, matching `linear`'s own
zero-initialized weight/bias. `output = gate * (W@action + b)` is a product of two independently-
zero quantities — by the product rule, every gradient (`d(gate)`, `d(W)`, `d(b)`) is exactly zero,
forever, a permanent fixed point discovered by loading trained checkpoints and finding
`external_cond`'s output byte-identical for real/random/zero actions after thousands of steps, in
every fine-tuning run of both models. Fix: initialize `gate = torch.nn.Parameter(torch.ones(1))`
instead of zeros, keeping `linear` zero-initialized — output is still exactly 0 at step 0, but
`d(loss)/d(linear.weight) = gate * action != 0` from the first step.

**This is a different bug from WorldMem's own, stock `zero_init_gate` option**
(`baselines/WorldMem/experiments/exp_base.py:140`, confirmed via `git blame` to be original
upstream code from commit `5555c7b1`, dated 2025-05-27, never touched locally — PERSIST's fine-
tuning code doesn't even call into `exp_base.py`, it bypasses that file's Hydra-driven machinery
entirely). That option zeros a slice of a weight matrix whose *input* (a conditioning embedding)
is never itself zero, so its gradient is nonzero from step 1 — structurally safe. PERSIST's
bolted-on `GatedActionEmbed` multiplied two independently-zero factors, which was never safe to
combine — a genuinely different failure mode, not a duplicate report of the same upstream issue.

### Three-group optimizer, warmup, freeze/unfreeze

Every trainable parameter is partitioned exactly once into three groups:

```python
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": args.lr_pretrained},   # default 1e-6 -- everything else
    {"params": pose_params,     "lr": args.lr_pose},          # default 1e-5 -- pose_cond_mlp,
                                                                #   temporal_pose_cond_mlp,
                                                                #   r_adaLN_modulation params
    {"params": new_layer_params, "lr": args.lr_new_layer},    # default 1e-4 -- the action layer
])
```

`--freeze-backbone` sets `requires_grad_(False)` on all `pretrained_params`, training only the new
action layer; `--unfreeze-adaln` additionally re-enables every param whose name contains
`"adaLN_modulation"`. This mirrors the same targeted-adaptation strategy that fixed Oasis's
fine-tuning instability (training the full backbone caused severe striped-corruption rollouts on
both models; freezing most of it and training only the action layer +/- adaLN avoids that).

Linear warmup, no decay, applied uniformly across all param groups:
```python
def lr_lambda(step): return min(1.0, (step + 1) / args.warmup_steps)   # default warmup_steps=200
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

**Why `lr_pose` exists as its own group**: pose-conditioning params are inherited from
pretraining (not new, unlike the action layer) but were previously bundled into the very slow
`lr_pretrained` group. Motivating finding: a real/shuffled/zero pose ablation on two checkpoints
found feeding the model its own real, correct pose consistently scored *worse* than wrong or zero
pose — ruled out eval-methodology gaps, a geometric FOV bug, and fine-tuning-induced corruption
(&lt;2% relative weight change even at `lr_pretrained=1e-6`) — leaving "pose conditioning needs
Craftium-specific recalibration but was never given the chance" as the remaining explanation. This
flag exists to test that hypothesis with a faster-moving LR on just those params.

### Three action-layout modes (mutually exclusive, asserted so in both `finetune_worldmem.py` and `finetune_worldmem_data.py`)

- **Default**: Craftium's raw 23-dim boolean action vector (`CRAFTIUM_ACTION_DIM = 23`).
- **`--use-clean-camera-features`**: appends 2 columns (`yaw_delta`, `pitch_delta`, degrees,
  gated on each axis's own mouse action being active) → 25-dim (`CRAFTIUM_ACTION_DIM_ENRICHED`).
- **`--use-worldmem-native-action-layout`**: reconstructs directly into WorldMem's own 25-slot
  `ACTION_KEYS` column layout, camera as **sign-only** `{-1,0,1}` → 25-dim
  (`CRAFTIUM_ACTION_DIM_WORLDMEM_NATIVE`). See the full column-mapping table below.

`--informed-action-init` copies the pretrained checkpoint's own weight *columns* for the handful
of dims that actually received training signal during WorldMem's own pretraining, into the new
layer's corresponding columns (bias copied in full, everything else stays zero). For the native
layout this is a trivial identity map (`{i: i}`, see below) since the native layout already places
every value at WorldMem's own column position.

## 2. `baselines/finetune_worldmem_data.py` — `CraftiumWorldMemDataset` vs. Oasis's `CraftiumOasisDataset`

```python
DATASET_ROOT = (
    "/scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/"
    "snapshots/d3a3bee6560304397093ae66523451b9e0d067f7"
)
```
(Fixed this session — was pointing at a stale `/home/lsokoudj/...` prefix; repoint to wherever the
`PERSIST-team/persist-eval-sample` HF snapshot lives on the new server.)

| | Oasis's `CraftiumOasisDataset` | WorldMem's `CraftiumWorldMemDataset` |
|---|---|---|
| Output | `(frames, actions)` | `(video, actions, poses, timestamp)` — WorldMem needs pose + frame-index conditioning |
| Windowing | dense "sliding" pool (every start position) or "fixed" (one/episode) | fixed non-overlapping grid: `windows_per_episode = 600 // n_frames` |
| Memory frames | none | each window also carries `memory_condition_length` (default 8) extra frames |
| Action shift | shifted by one frame ("action that *produced* frame t"), frame 0 = zero action | **no shift** — read at the same indices as `video`, matching WorldMem's own convention |
| Pose | none | derived from `cam_dir`, not `player_yaw`/`player_pitch` |

**Two fidelity-critical fixes baked into this file, both essential to replicate:**

1. **Pose source**: `pitch_yaw_from_cam_dir` derives `(pitch, yaw)` from Craftium's `cam_dir`
   ground-truth 3D look vector — **not** `player_pitch`/`player_yaw`, which were found frozen
   96%/62% of the time on real data. The formula is the algebraic inverse of WorldMem's own
   `df_video.py::euler_to_rotation_matrix`, verified to round-trip with 0.0 error.
   `normalize_pose` subtracts the window's first-frame position and wraps angles to `[0,360)` —
   does **not** negate yaw (an earlier version did, to compensate for the broken `player_yaw`
   field; no longer needed once deriving from `cam_dir`).

2. **Memory-frame selection**: `_select_memory_indices` is an exact port of WorldMem's own
   `df_video.py::_generate_condition_indices` — the real greedy FOV-overlap+recency matcher
   actually used at generation time, including the "remove already-covered FOV region before
   picking the next" step. All numeric constants (`num_samples=10000`, `radius=30`, `±105°`/`±75°`
   FOV half-angles, `-0.2` recency weight) are copied verbatim from upstream. Note: `radius` was
   briefly rescaled to `1.5` in an earlier attempt (Craftium camera displacement over 30 frames
   measured ~0.88 units) but a full re-fine-tune with that change produced a *worse*, inverted
   causality ordering — reverted to `radius=30` to match upstream exactly. Earlier versions also
   used pure random selection, then a *different* wrong function borrowed from
   `MinecraftVideoDataset.load_data`'s distance-threshold heuristic — both were train/inference
   mismatches versus what generation actually does; only the current `_generate_condition_indices`
   port is correct.

## 3. `baselines/craftium_action_features.py` — the WorldMem-specific half

This module is shared with the Oasis pipeline. Everything from `WORLDMEM_PRETRAINING_REACHABLE_SLOTS`
onward exists purely for WorldMem.

```python
# WorldMem's own convert_action_space() (minecraft_video_dataset.py:46-56) only ever writes 8 of
# ACTION_KEYS' 25 slots during pretraining data generation: hotbar.1(2), forward(11), back(12),
# left(13), right(14), cameraY(15), cameraX(16), drop(24). The other 17 slots' pretrained weight
# columns never received a real training signal (always-zero input throughout pretraining), so
# this reconstruction keeps them forced to 0 even where Craftium *could* supply a real value.
WORLDMEM_PRETRAINING_REACHABLE_SLOTS = np.zeros(25, dtype=bool)
WORLDMEM_PRETRAINING_REACHABLE_SLOTS[[2, 11, 12, 13, 14, 15, 16, 24]] = True

CRAFTIUM_ACTION_DIM_WORLDMEM_NATIVE = 25

def augment_actions_to_worldmem_native_layout(action, player_yaw, player_pitch):
    n = action.shape[0]
    vec = np.zeros((n, 25), dtype=np.float32)
    vec[:, 11] = action[:, 0]   # forward   <- Craftium forward
    vec[:, 12] = action[:, 1]   # back      <- Craftium backward
    vec[:, 13] = action[:, 2]   # left      <- Craftium left
    vec[:, 14] = action[:, 3]   # right     <- Craftium right
    vec[:, 2]  = action[:, 10]  # hotbar.1  <- Craftium slot_1 (place-coupled caveat, see below)
    cam = compute_clean_camera_deltas(player_yaw, player_pitch, action)  # (T,2) = [yaw_delta, pitch_delta]
    vec[:, 16] = np.sign(cam[:, 0])  # cameraX <- sign(yaw_delta)   -- {-1,0,1} ONLY, not magnitude
    vec[:, 15] = np.sign(cam[:, 1])  # cameraY <- sign(pitch_delta) -- {-1,0,1} ONLY, not magnitude
    # drop(24) left 0.0 -- unreachable on both sides, nothing to map
    vec[:, ~WORLDMEM_PRETRAINING_REACHABLE_SLOTS] = 0.0  # defensive re-assertion
    return vec

WORLDMEM_NATIVE_INFORMED_INIT_ACTION_MAP = {i: i for i in [2, 11, 12, 13, 14, 15, 16, 24]}
```

**Column-mapping table (Craftium source idx → WorldMem `ACTION_KEYS` dest idx):**

| Craftium idx | meaning | → WorldMem idx | meaning | note |
|---|---|---|---|---|
| 0 | forward | 11 | forward | direct 1:1, both pure bool |
| 1 | backward | 12 | back | direct 1:1 |
| 2 | left | 13 | left | direct 1:1 |
| 3 | right | 14 | right | direct 1:1 |
| 10 | slot_1 | 2 | hotbar.1 | closest match; Craftium's slot_1 is only ever set jointly with `place`(8) — kept anyway, almost always 0 in practice |
| derived `sign(yaw_delta)`, gated on action[19]/[20] | mouse_x | 16 | cameraX | **sign-only**, see below |
| derived `sign(pitch_delta)`, gated on action[21]/[22] | mouse_y | 15 | cameraY | **sign-only**; note Y=15/X=16 — opposite of Oasis's ordering |
| — | — | 24 | drop | forced 0.0 — unreachable on both sides |
| (jump, sneak, aux1/sprint, dig/attack, slot_2-9, inventory, ESC, swapHands, use, pickItem) | — | forced 0 via reachable-slots mask | — | WorldMem's own pretraining never wrote real signal here — uninformative columns, explicit controlled-variable exclusion |

**Why sign-only, not magnitude (the key divergence from Oasis's mapping in the same file):**
verified directly against `minecraft_video_dataset.py:48-51`'s `convert_action_space`, WorldMem's
own pretraining data never produces any camera value other than exactly `-1`, `0`, or `1` — so the
pretrained weight columns were calibrated for that exact magnitude. Feeding real yaw/pitch degree
deltas (which can be 6° or larger) would be several times outside that calibrated range, actively
counterproductive for informed-init. Oasis's own pretraining (`open-oasis/utils.py`'s
`one_hot_actions`), by contrast, used a continuous 40-bucket magnitude encoding, so Oasis's native
mapping in this same file computes a real scaled value instead:
`np.clip(cam[:,0] / OASIS_CAMERA_BIN_SIZE / OASIS_CAMERA_NUM_BUCKETS, -1.0, 1.0)`. **Do not copy
Oasis's magnitude approach when working on the WorldMem side, and do not copy WorldMem's
sign-only approach when working on the Oasis side** — this was a deliberate, verified-from-source
divergence, not an inconsistency to "fix."

A second WorldMem-specific gotcha: `ACTION_KEYS` orders `cameraY=15, cameraX=16` — the opposite
of Oasis's `cameraX=15, cameraY=16`. Documented further in `docs/action_space_comparison.md`,
which also flags a same-repo inconsistency in WorldMem's own upstream code between
`models/utils.py` (orders cameraX/cameraY) and `minecraft_video_dataset.py` (orders
cameraY/cameraX) — the latter is the one that matters at runtime, since its `convert_action_space`
is what actually produced the pretraining data.

## 4. The generation-time bug fix — `baselines/finetune_worldmem_generate.py`

`build_worldmem_shell` overrides one config field after loading the stock yaml:
```python
cfg.chunk_size = cfg.n_tokens   # stock default: 1
```
**Root cause**: `validation_step`'s sliding window re-invokes `_prepare_conditions` every new
chunk, recomputing each in-window frame's relative pose conditioning from scratch using whatever
memory-frame set `_generate_condition_indices` picks for the *current* step — including frames
from prior chunks still in the window whenever `chunk_size < n_tokens`. Already-generated frames
are held at noise level 0 (pixels frozen) but the model still attends to them under
newly-inconsistent conditioning, producing severe structured pixel corruption starting exactly at
frame `memory_condition_length` — reproduced on both untrained and fine-tuned checkpoints. Setting
`chunk_size = n_tokens` makes `start_frame == curr_frame` every step (zero window overlap),
eliminating it. Purely a generation-time setting (`chunk_size` is only read inside
`validation_step`, never `training_step`) — **no retraining was needed**, and this fix is baked
into `build_worldmem_shell`, so every downstream script that imports it (all 4 eval/grid scripts
below) inherits the fix automatically. A from-scratch reimplementation that skips this shell
function must replicate that one line or will reproduce the corruption bug.

## 5. Remaining scripts, one line each

- **`eval_worldmem_heldout_loss.py`** — cheapest sanity check: `training_step`'s loss (stock
  method), before-vs-after checkpoint, fixed held-out batch, identical seed both times.
- **`eval_worldmem_finetuned_fvd.py`** — FVD + PSNR over train/test split, 3 rollout-length
  truncations (200/400/600 frames). Depends on `build_worldmem_shell` (chunk_size fix included).
- **`eval_worldmem_pose_causality.py`** — real/shuffled/zero pose ablation (actions held fixed),
  single checkpoint/episode. Depends on `build_worldmem_shell`.
- **`eval_worldmem_pose_causality_cleancam.py`** — extended version for 25-dim checkpoints, adds
  `--num-seeds` (repeat-measurement noise check) and `--ablate {pose,action}` with
  `--action-layout {bespoke,native}` (ablates the correct camera columns per layout: `[23,24]`
  bespoke, `[15,16]` native).
- **`eval_worldmem_zeroshot.py`** — zero-shot baseline, **no** fine-tuning or action-dim surgery
  at all: original 25-dim checkpoint via stock `experiments.exp_base.load_custom_checkpoint`,
  synthetic WASD key pattern, FVD against Craftium seed frames. This is the "before" baseline
  the fine-tuning in §1 is meant to improve on.
- **`zeroshot_minecraft_worldmem.py`** — true zero-shot on real Minecraft clips (not Craftium at
  all), original checkpoint, original `interactive()` call sequence. Only adaptation: a
  camera-index swap constant to reinterpret Oasis's own recorded VPT actions in WorldMem's
  convention. Independent of everything above.
- **`make_comparison_grid_worldmem.py`** — per-episode `.npz` comparison data (real /
  untrained-surgery / fine-tuned), stitched later with Oasis's equivalent grid. Depends on
  `build_worldmem_shell`.
- **`worldmem_validate_example.py`** — earliest de-risking script, pure upstream, no
  fine-tuning, confirms `interactive()`'s call sequence works outside the Gradio UI using a
  Craftium seed frame (WorldMem's own bundled examples are inaccessible Git-LFS pointers).
  First discovered that sustained hard camera-turn sequences degrade output quality.

## Known open item — not fixed, flagged for the new server

No `.slurm`/launch script exists anywhere in this repo for WorldMem fine-tuning (unlike Oasis,
which has several `run_*.slurm` examples) — WorldMem fine-tuning has apparently never been run
end-to-end on *this* server via a tracked, reproducible launch path. You'll need to write one from
scratch using `finetune_worldmem.py`'s `Args` dataclass CLI flags as reference. One placement
detail: `config_path`'s default (`configurations/huggingface.yaml`) is relative, so launch scripts
need `cd baselines/WorldMem` before invoking `finetune_worldmem.py` (which itself lives one level
up, in `baselines/`) — or pass an absolute `--config-path`.
