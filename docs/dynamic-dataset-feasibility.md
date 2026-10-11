# Can PERSIST train on a dynamic dataset (moving entities) with the current code?

## Context

PERSIST has so far only been trained on "static" datasets — the player agent is the only moving
thing in the environment, everything else (terrain) is inert. The user is generating a "dynamic"
dataset on another server (mobs like sheep visible in both the voxel grid and the video), which
will be made available to this repo once ready. Before that happens, the user wants to know
whether the **current, unmodified PERSIST code** can learn from it, or whether something
(architecture or otherwise) needs to change first.

## What research into this codebase found

Two investigations (one into the data-generation pipeline, one into the model/training
architecture) established the following, with exact file:line citations kept here for reference:

**The one real structural blocker, as shipped in this repo**: the default voxel-extraction path
(`gym_envs/craftium/src/script/scripting_server.cpp:46-112`, `get_voxel_data_cpp`) reads only
Luanti's MapNode (terrain) layer — mobs are a separate ActiveObject/entity system that never
reaches `obs_voxel_mt`. The voxel class vocabulary
(`dataset_toolkits/default_mt_voxel_classdict.json`, ~2138 classes) and the voxel VAE's one-hot
encoder (`models/vae_voxel.py:194-229`, `in_channels=2138`) are built purely from terrain
material types, with no entity slot. **The user has already modified their data-generation setup
on the other server specifically to solve this** — their voxel grid and video will both show mobs
moving — so this blocker does not apply to their dataset as-is. What remains open is whether their
modification is compatible with this repo's current code shape, which is what the checklist below
resolves.

**No other architectural blocker was found.** The conditioning signal
(`models/embeddings.py:18-49`, `ActionCameraEmbedder`) is a single global per-frame vector from
the player's own action + camera pose, with no per-entity input anywhere — but the user's stated
goal is PERSIST *predicting* other-object motion, not *controlling* it, and nothing about that
goal requires new conditioning inputs. The voxel DiT's temporal self-attention
(`models/dit_voxel.py:144-208`) is the same mechanism that already has to learn persistence of the
player's own terrain edits over time; in principle it's equally capable of learning temporal
coherence for entity-class voxels, provided they're actually present in the training data (which
they now will be). Nothing found suggests the DiT/denoiser code itself needs to change.

A secondary, non-blocking finding: `dataset_toolkits/process_mt_data.py:402-456`'s
`compute_importance_masks` models frame-to-frame voxel change as one global camera-translation +
residual — a heuristic tuned for a single-mover world, not a hard architectural wall, but relevant
now that entity motion will show up as extra voxel changes (see checklist item 4).

## Checklist to verify on the other server before training

The user wasn't certain about several implementation details of their own modification — these
are the open items to check, each tied to what it determines:

1. **Voxel class vocabulary size** — count the entries in the dynamic dataset's classdict JSON
   (the equivalent of `dataset_toolkits/default_mt_voxel_classdict.json`) and compare to 2138. If
   larger, `models/vae_voxel.py`'s `in_channels=2138` must change before the VAE architecture can
   accept the data at all.
2. **Raw voxel data shape** — confirm `obs_voxel_mt` is still `(T, D, D, D, 2)` (entities added as
   new `(node_id, node_param)` combinations within the existing format), not a new separate field.
   A new field would require changes to `data_loaders/minetest_latent_camera_action_dataset.py`
   beyond just a channel-count change.
3. **Node-id collisions** — verify the `node_id` values used for entities don't collide with real
   terrain content ids already in use (the default pipeline reserves `126`/`127` for air/ignore —
   `EMPTY_SPACE_NODE_IDS` in `dataset_toolkits/generate_raw_data.py:25`).
4. **Frame-filtering threshold** — check `process_mt_data.py`'s `compute_importance_masks` /
   `max_vox_per_ts = 40` and whether `remove_bad_frames` (default `True` in
   `train_scripts/train_diffuser_voxel.py`) would discard frames where a mob's motion exceeds 40
   voxels of change not explained by a single camera translation. May need raising the threshold
   or disabling the filter for this dataset once real per-frame entity-motion voxel counts are
   known.
5. **Entity identity/persistence** — not something to fix, just to know going in: nothing in
   PERSIST's architecture tracks individual object identity across frames. Entities will be
   learned as a voxel *class* moving through the grid, the same way terrain edits are learned —
   plausible, not guaranteed, and the model has no explicit protection against e.g. two entities
   near each other being confused.

## Known cost regardless of checklist outcome

The existing pretrained voxel VAE checkpoint was trained on a vocabulary that never saw entity
classes. Even if vocabulary size turns out unchanged (item 1), its weights don't encode the new
entity semantics — the voxel VAE needs real retraining on the dynamic dataset (not a fine-tune
resume from the current checkpoint), and the voxel denoiser downstream needs retraining after that
too, since it operates in the VAE's latent space.

## Next step

User is going to check items 1-4 against their dataset-generation setup on the other server, then
come back with answers. Once known, a concrete implementation plan (code changes needed, if any,
plus the retraining sequence: voxel VAE first, then voxel denoiser, camera model and pixel
pipeline as needed) can be written.
