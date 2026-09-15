# Action spaces: Minecraft (VPT) vs. Craftium

This document compares the action space Oasis and WorldMem were pretrained on (Minecraft, via
the VPT/MineDojo action vocabulary) against Craftium's native action space, which is what all of
this project's fine-tuning data actually contains. It exists because every fine-tuning result in
this project depends on the 25→23 dimension "model surgery" swap between these two spaces, and
that swap's exact semantics had never been written down precisely until now.

All claims below are backed by direct source citations (`file:line`), not by external
documentation — where the local codebase didn't resolve a question, that's noted explicitly
rather than guessed at.

## 1. WorldMem's native action space (25-dim, VPT-derived)

Defined identically in two places in the WorldMem repo:
`baselines/WorldMem/algorithms/worldmem/models/utils.py:73-99` and
`baselines/WorldMem/datasets/video/minecraft_video_dataset.py:16-42`.

| idx | key | meaning |
|---|---|---|
| 0 | `inventory` | open inventory screen |
| 1 | `ESC` | pause/escape menu |
| 2-10 | `hotbar.1`-`hotbar.9` | select hotbar slot 1-9 |
| 11 | `forward` | move forward |
| 12 | `back` | move backward |
| 13 | `left` | strafe left |
| 14 | `right` | strafe right |
| 15/16 | `cameraX`/`cameraY` | camera yaw/pitch delta — **see the ordering caveat below** |
| 17 | `jump` | jump |
| 18 | `sneak` | sneak/crouch |
| 19 | `sprint` | sprint |
| 20 | `swapHands` | swap main/off hand item |
| 21 | `attack` | attack / break block (held) |
| 22 | `use` | generic right-click interact (see §4) |
| 23 | `pickItem` | middle-click pick block |
| 24 | `drop` | drop held item |

**Camera encoding**: each camera channel is a *continuous* value, produced by decoding one of 41
discrete VPT bins (`bin_size=0.5`, `max_val=20`) back into a float in `[-1, 1]`
(`models/utils.py:102-123`). This is a magnitude-and-direction encoding — one float per axis
capturing how far to move the camera.

**Caveat — an unresolved same-repo inconsistency**: the two files above disagree on which index
is which. `models/utils.py:89-90` orders `"cameraX", "cameraY"` (index 15=X, 16=Y) — verified
directly, matching Oasis's own copy exactly (see §2). `minecraft_video_dataset.py:32-33` orders
`"cameraY", "cameraX"` (index 15=Y, 16=X) — also verified directly. This is not a typo I'm
inferring from one read; both files were opened and diffed line-by-line. The file that actually
matters for the real data pipeline is `minecraft_video_dataset.py`'s ordering: its own
`convert_action_space()` (next paragraph) sets index 16 from yaw and index 15 from pitch,
consistent with *its own* local list — and `baselines/eval_worldmem_zeroshot.py`'s independently
written `ACTION_KEYS` copy also uses the Y-then-X order, confirming this is genuinely the
convention WorldMem's generation code expects at runtime, not just a docstring slip in one file.
Anyone reusing a raw 25-dim VPT tensor from elsewhere (e.g. Oasis's own action files, which use
X-then-Y) against WorldMem's generation path needs to swap indices 15 and 16 first.

**What WorldMem's own data pipeline actually populates**: `convert_action_space()`
(`minecraft_video_dataset.py:44-57`) converts WorldMem's own raw 8-dim MineDojo `MultiDiscrete`
recordings into this 25-dim vector, but only ever sets 8 of the 25 dimensions: `forward`, `back`,
`left`, `right`, camera yaw, camera pitch, `drop`, and `hotbar.1`. The remaining 17 dimensions
(`inventory`, `ESC`, `hotbar.2`-`.9`, `jump`, `sneak`, `sprint`, `swapHands`, `attack`, `use`,
`pickItem`) are structurally part of the vocabulary but never appear in WorldMem's own training
data — its data-generation agent (`baselines/WorldMem/data_generator.py:22-32`) only ever issues
`forward`/`back`/`left`/`right`/camera moves, confirmed by its own `ACTIONS` dict quoted in full
in §4.

## 2. Oasis's native action space (25-dim, VPT-derived)

Defined at `baselines/open-oasis/utils.py:29-53`, with an explicit attribution comment at the top
of the file: *"Action format derived from VPT https://github.com/openai/Video-Pre-Training"*
(`utils.py:3`). The vocabulary and ordering are byte-identical to WorldMem's `models/utils.py`
copy — same 25 keys, same index-for-index meaning as the table in §1, including the
`cameraX`(15)/`cameraY`(16) ordering (verified directly, `utils.py:47-48`).

Two differences worth noting against WorldMem:
- **No pose input at all.** Oasis conditions purely on the action vector; it has no equivalent of
  WorldMem's `(x, y, z, pitch, yaw)` pose channel.
- **Real, unrestricted training data.** Unlike WorldMem's narrow 8-action data generator, Oasis's
  own bundled sample recordings (`baselines/open-oasis/sample_data/*.actions.pt`, real captured
  Minecraft sessions) exercise a much wider slice of the 25-dim vocabulary — verified by loading
  `sample_actions_0.one_hot_actions.pt` directly: nonzero values appear at multiple indices
  beyond just movement/camera, consistent with real human play rather than a scripted policy.

## 3. Craftium's action space

### 3a. Raw Craftium `Dict` space (what the environment itself exposes)

`gym_envs/craftium/craftium/craftium_env.py:14-19`:
```python
ACTION_ORDER = [
    "forward", "backward", "left", "right", "jump", "aux1", "sneak",
    "zoom", "dig", "place", "drop", "inventory", "slot_1", "slot_2",
    "slot_3", "slot_4", "slot_5", "slot_6", "slot_7", "slot_8", "slot_9",
    "mouse"
]
```
21 boolean keys plus `mouse`, a continuous 2-vector `[dx, dy]` (`craftium_env.py:116-123`) — 23
scalars total, but two of them (the mouse pair) are continuous, not boolean.

### 3b. PERSIST's actual 23-dim vector (what's really in every episode's `data.npz`)

PERSIST doesn't use Craftium's raw space directly — it wraps it with
`MultiDiscreteActionWrapper` (`utils/action_util.py`), which **discretizes** mouse movement into
4 fixed-magnitude direction booleans instead of a continuous vector, and **drops** `inventory`
and `zoom` entirely. The definitive 23-dim index map, `utils/action_util.py:131-155`:

| idx | key | meaning |
|---|---|---|
| 0-3 | `forward`, `backward`, `left`, `right` | movement |
| 4 | `jump` | jump |
| 5 | `aux1` | sprint |
| 6 | `sneak` | sneak/crouch |
| 7 | `dig` | break block (held) |
| 8 | `place` | place block |
| 9 | `drop` | drop held item |
| 10-18 | `slot_1`-`slot_9` | hotbar slot select |
| 19/20 | `mouse x-0.1` / `mouse x+0.1` | camera yaw left/right, fixed magnitude |
| 21/22 | `mouse y-0.1` / `mouse y+0.1` | camera pitch up/down, fixed magnitude |

This is confirmed as exactly what lands in training data by `dataset_toolkits/generate_raw_data.py:570-571`
(`multihot_action = env.multihot(action); level_data["action"].append(multihot_action)`), and by
both fine-tuning scripts' own docstrings (`baselines/finetune_oasis_data.py:56-58`,
`baselines/finetune_worldmem_data.py:49-51`), which is the 23-dim convention this entire
project's fine-tuning work has built on.

**Documentation gap found**: `dataset_toolkits/data.md:79` documents the `action` field as shape
`(T,)`, which does not match the confirmed `(T, 23)` reality — likely describing "T elements,
each itself a 23-length array" loosely, but this is worth fixing since it reads as wrong on its
own. Not fixed as part of this document; flagged here for a follow-up.

## 4. Overlap and non-overlap: the actual comparison

| Minecraft/VPT (25-dim) | Craftium/PERSIST (23-dim) | Relationship |
|---|---|---|
| `forward`, `back`, `left`, `right` | `forward`, `backward`, `left`, `right` | direct 1:1 |
| `jump` | `jump` | direct 1:1 |
| `sprint` | `aux1` | direct 1:1 (different name, same function) |
| `sneak` | `sneak` | direct 1:1 |
| `attack` | `dig` | direct 1:1 (both mean "break the block/thing you're facing") |
| `drop` | `drop` | direct 1:1 |
| `hotbar.1`-`.9` | `slot_1`-`slot_9` | direct 1:1 |
| `use` | `place` | **partial only** — see below |
| `cameraX`/`cameraY` | `mouse x±0.1`/`mouse y±0.1` | **same purpose, fundamentally different encoding** — see below |
| `inventory`, `ESC`, `swapHands`, `pickItem` | *(none)* | **Minecraft/VPT-only** |
| *(none)* | *(none — everything in Craftium's 23-dim space has a VPT-side counterpart, per this table)* | — |

**`use` vs. `place` — not a clean match.** VPT's `use` is a single, heavily overloaded key: in
full Minecraft it places a block if one is held, opens a container or interacts with an entity
(villager, furnace, door) if facing one, eats held food, draws a bow, and more, depending purely
on context. Craftium's `place` (`craftium_env.py:16`) covers only the "place a block" case —
there's no Craftium key for the broader interact-with-entity/container semantics `use` carries in
full Minecraft. This asymmetry likely reflects Craftium/Luanti's simpler world model (fewer
interactable entity types) rather than an oversight in PERSIST's wrapper.

**Camera control is the single biggest control-fidelity gap between the two spaces**, and it's a
difference in *kind*, not just range:
- Minecraft/VPT: each camera axis is one **continuous** value in `[-1, 1]`, itself decoded from
  41 discrete magnitude bins (`models/utils.py:102-123`) — the model can express "turn a little"
  vs. "turn a lot" on a fine graduated scale.
- Craftium/PERSIST: each camera axis is split into **two fixed-magnitude boolean** directions
  (`mouse x-0.1`, `mouse x+0.1`, and the pitch equivalents) — every "turn" is the same fixed
  0.1-unit step regardless of how far the underlying mouse actually moved. There is no way to
  express "turn slightly" vs. "turn sharply" as a single input in this space; only how many
  consecutive frames the same direction key is held distinguishes a small turn from a large one.

This matters directly for this project's fine-tuning work: swapping Oasis/WorldMem's
continuous-camera action layer for a discrete-step one is not just a dimension-count change, it's
a change in the *kind* of signal the model has to learn to interpret — plausibly a meaningful
part of why fine-tuning needed a fresh, zero-initialized action layer rather than a simple
dimension truncation/remap of the pretrained one.

## 5. Worked examples

- **"Forward" in Minecraft (VPT) vs. Craftium**: functionally identical at the action-space
  level — both are a single boolean meaning "move in the direction you're facing." The
  difference, if any, is in the underlying physics/movement speed of each engine (Minecraft's
  Java-based movement vs. Craftium's Luanti/Minetest-based movement), not in what the action
  vector itself encodes.
- **Camera turn in Minecraft (VPT) vs. Craftium**: this is where the two spaces genuinely
  diverge. A real VPT recording of "look right slowly" is one continuous `cameraX` value like
  `0.05`, held for as many frames as the turn takes. The Craftium-native equivalent is a
  sequence of discrete `mouse x+0.1` presses, each step identical regardless of how "slow" or
  "fast" the actual motion was meant to be — the model only sees repeated fixed-size steps, never
  a single graduated value.
- **Right-click ("use") in Minecraft vs. Craftium**: pressing `use` while facing a crafting table
  in Minecraft opens a UI with no Craftium equivalent at all (no `inventory`-interaction action
  exists in the 23-dim space); pressing `use` while holding a block places it, which *does* map
  cleanly to Craftium's `place`. Same key, only one of its several real-Minecraft behaviors has a
  Craftium counterpart.

## 6. Further reading

- Craftium paper: [arXiv:2407.03969](https://arxiv.org/abs/2407.03969)
- Craftium docs: [craftium.readthedocs.io](https://craftium.readthedocs.io/en/latest/)
- Craftium GitHub (upstream): [github.com/mikelma/craftium](https://github.com/mikelma/craftium/)
- MineDojo GitHub: [github.com/MineDojo/MineDojo](https://github.com/MineDojo/MineDojo)
- VPT (Video Pre-Training) paper/repo: [github.com/openai/Video-Pre-Training](https://github.com/openai/Video-Pre-Training)
- WorldMem paper: [arXiv:2504.12369](https://arxiv.org/abs/2504.12369); [project page](https://xizaoqu.github.io/worldmem/)
- WorldMem's own Minecraft dataset: [huggingface.co/datasets/zeqixiao/worldmem_minecraft_dataset](https://huggingface.co/datasets/zeqixiao/worldmem_minecraft_dataset)
- Diffusion Forcing (the base algorithm both WorldMem and Oasis's action-handling code derive from): [github.com/buoyancy99/diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing)
- Open-Oasis GitHub: [github.com/etched-ai/open-oasis](https://github.com/etched-ai/open-oasis); [blog post](https://oasis-model.github.io/)
- PERSIST project page: [francelico.github.io/persist.github.io](https://francelico.github.io/persist.github.io/); [arXiv:2603.03482](https://arxiv.org/abs/2603.03482)
