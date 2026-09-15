# Action-space empirical comparison: Craftium vs. WorldMem's Minecraft dataset vs. Oasis's Minecraft dataset

Companion to `docs/action_space_comparison.md` (which documents the *vocabulary/format*
differences from source code). This report measures the *actual, real-recorded data* directly:
which action dims are genuinely exercised, how often, and what real-world magnitude each
produces — using files already downloaded in this project, not the vocabulary spec alone.

Script: `scratchpad/action_space_empirical_analysis.py` (full output below, condensed).

## 1. Which action dims does each dataset actually use? (answers "did the original code use all 25 dims")

| Dataset | Dims ever nonzero | Which ones |
|---|---|---|
| WorldMem's own dataset (`ep0_desert`, `ep1_plains`, 150 frames each) | **4 / 25** | forward, back, cameraY, cameraX |
| Oasis's own dataset, `Player729` clip (1200 frames) | **15 / 25** | hotbar.2-7, forward, back, left, right, camera, jump, attack, use |
| Oasis's own dataset, `snippy-chartreuse-mastiff` clip (1200 frames) | **19 / 25** | inventory, hotbar.1-9, forward/back/left/right, camera, sneak, attack, use |
| Oasis's own dataset, `treechop` clip (1200 frames) | **7 / 25** | hotbar.3-5, forward, camera, attack |
| Craftium (20 sampled train episodes, 12,000 frames) | **11 / 23** | forward, backward, left, right, jump, aux1, sneak, mouse_x-/x+, mouse_y-/y+ |

**Answer to "did the original code use all 25 dims": no, neither model's own training/eval data
comes close to exercising the full vocabulary in any single episode** — but the *reason* differs
sharply between the two:

- **WorldMem's own data is narrow by design.** Only forward/back/camera ever appear in these
  episodes — no jump, no attack, no hotbar, no inventory. This matches `docs/action_space_comparison.md`'s
  finding that WorldMem's own data-generator agent is a small scripted policy, not human play.
- **Oasis's own data is rich because it's real human recordings** — 7-19 distinct actions per
  clip, including hotbar switches, attack, use, sneak, inventory — genuine human Minecraft
  sessions, not a scripted policy.
- **Craftium's own data sits in between but has its own gap**: dig, place, drop, and all 9 hotbar
  slots are used **zero times** across all 12,000 sampled frames. This directly affects the
  informed-init idea (see below).

## 2. Craftium vs. WorldMem's Minecraft dataset

**Format**: different vocabularies (23 vs. 25 dims), mapped in `docs/action_space_comparison.md`
section 4. No shared raw index numbering — any comparison requires the explicit mapping table.

**Real per-action magnitude** (does "forward"/"camera turn" mean a different amount of real
motion in each engine?):

| | WorldMem `ep0_desert` | WorldMem `ep1_plains` | Craftium `level_001` | Craftium `level_003` |
|---|---|---|---|---|
| position-delta, forward ON | 0.185 | 0.156 | 0.120 | 0.145 |
| position-delta, forward OFF | 0.066 | 0.106 | 0.059 | 0.038 |
| yaw-delta (deg), camera ON | 11.75 | 12.13 | 13.48 | 8.13 |
| yaw-delta (deg), camera OFF | 0.00 | 0.00 | 0.23 | 0.16 |

**Finding, and it cuts against the "different magnitude" hypothesis**: the actual per-step
magnitudes are in the **same order of magnitude** for both forward-movement (~0.04-0.19 units)
and camera-turn (~8-13.5 degrees) — not the "2 boxes vs. 1 box" scale mismatch that seemed like
the likely culprit going in. This doesn't rule out scale mismatch as *a* contributing factor, but
it means it's not a dramatic, obvious one for these two specific actions in these sample episodes.

**One real anomaly worth flagging, not smoothing over**: in both datasets, meaningful position
change still happens even when "forward" is marked OFF (0.066-0.106 for WorldMem, 0.038-0.059
for Craftium) — non-trivial relative to the ON case. Likely causes: other movement keys
(back/left/right) also displacing the player, physics (falling/momentum), or an off-by-one
alignment between an action's index and the frame transition it actually caused. This wasn't
resolved here and is worth a follow-up before leaning on these numbers too heavily.

**Implication for the informed-init idea (already implemented)**: only `forward`, `backward`,
`left`, `right` will ever receive a real training signal in Craftium fine-tuning — `drop` and
`slot_1` (hotbar.1), the other two dims in the informed-init mapping, are used **zero times** in
this Craftium sample, so copying their pretrained weights is harmless but likely inert in
practice. The informed init's real value is concentrated in the 4 movement dims.

## 3. Oasis's Minecraft dataset vs. WorldMem's Minecraft dataset

**Format**: nominally the same VPT vocabulary, same 25 keys, same continuous-per-axis camera
encoding — no format mismatch in principle. One real ordering gotcha already documented
(`docs/action_space_comparison.md` section 1): Oasis's own `utils.py` and WorldMem's *own*
`minecraft_video_dataset.py` disagree on whether index 15 is cameraX or cameraY — a swap this
project's zero-shot scripts already account for.

**The real, dominant difference is data realism/diversity, not format**:

| | WorldMem's own training data | Oasis's own sample clips |
|---|---|---|
| Source | Scripted 8-key policy (`data_generator.py`) | Real captured human Minecraft sessions |
| Dims ever active (this sample) | 4 (forward/back/camera only) | 7-19 per clip |
| Camera values | Present but from a scripted policy | Continuous, spans nearly the full [-1,1] range |
| Includes attack/use/inventory/hotbar/sneak? | No | Yes, all of them appear |

This is a clean, well-evidenced explanation for part of why WorldMem's own model generalizes
worse to *any* out-of-distribution action pattern (including Craftium's) than Oasis does: Oasis's
pretraining data was always more behaviorally diverse, independent of the Craftium question
entirely.

## 4. Bottom line

- Action **format** differences (dimension count, index order, continuous-vs-discrete camera)
  are real and already handled (model surgery, camera-index swaps).
- Action **magnitude** differences, tested empirically here, are surprisingly *not* the dominant
  factor for forward-movement or camera-turn specifically.
- Action **diversity/realism** is a genuine, substantial gap between WorldMem's own training data
  and Oasis's — worth remembering as a factor independent of Craftium.
- Craftium's own data never exercises `dig`/`place`/`drop`/any hotbar slot at all in this sample —
  worth double-checking against the full 236-episode set before concluding the model doesn't need
  to learn those at all.
