# Oasis, fully explained: architecture, shapes, training and inference

*Everything below was checked against the actual code in this repo:*
`baselines/open-oasis/{dit.py, attention.py, vae.py, utils.py, generate.py}` *(the released Oasis
model code)*, `baselines/finetune_oasis.py` *and* `baselines/finetune_oasis_data.py` *(our training
code)*, `baselines/craftium_action_features.py` *(our action translation)*, *the config of our run*
`outputs/finetune_oasis_luanti_hudnav_1000ep`, *and parameter counts read directly from the weight
files* `oasis500m.safetensors` *and* `vit-l-20.safetensors`.

---

## 0. How to read this document

**One running example everywhere.** Our fine-tuning run used 4 GPUs, each holding a batch of
**3 clips**, each clip being **32 consecutive frames**. So whenever you see a shape, imagine:

| Symbol | Meaning | Value in our run |
|---|---|---|
| **B** | batch size: how many independent clips are processed together on one GPU | 3 |
| **T** | time: how many consecutive frames in one clip | 32 |
| **C** | channels: how many numbers describe one spatial location | 3 (pixels: R,G,B) or 16 (latents) |
| **H, W** | height and width of a 2-D grid | 360 × 640 (pixels), 18 × 32 (latents), 9 × 16 (tokens) |
| **D** | hidden width: length of the vector that represents one token inside a transformer | 1024 |
| **h** | number of attention heads | 16 |
| **d** | width of one attention head (D / h) | 64 |

Shapes are always written **in order**, e.g. `[B, T, C, H, W] = [3, 32, 16, 18, 32]` means
"3 clips × 32 frames × 16 channels × 18 rows × 32 columns". Reading a shape left to right is like
zooming in: *which clip → which frame → which channel → which row → which column*.

---

## 1. The big picture in plain words

Oasis is a **world model**: you give it the last few frames of a game and the keys you are
pressing, and it draws the next frame. Repeat that forever and you are "playing" a game that only
exists inside a neural network. Oasis was released in 2024 by **Decart and Etched** as
*open-oasis* ("Oasis-500M"), trained on Minecraft gameplay.

It is made of **two separate neural networks** that do very different jobs:

1. **A video autoencoder (ViT-L/20, 229.2M weights, frozen).** Pixels are expensive: one 360×640
   color frame is 691,200 numbers. The autoencoder's **encoder** squeezes each frame into a small
   grid of codes (**16 × 18 × 32 = 9,216 numbers**, ~75× smaller), and its **decoder** turns codes
   back into pixels. It never changes during our training. Think of it as a very good "zip /
   unzip" for game frames.

2. **A Diffusion Transformer, "DiT" (607.9M weights).** This is the actual "brain". It never
   sees pixels; it works entirely on the small codes. It is a **diffusion model**: during
   training we take real codes, bury them in random noise, and teach the network to predict how
   to remove that noise. At play time it starts from pure noise for the next frame and removes the
   noise step by step, guided by the previous frames and the keys pressed.

Two small extra inputs steer the DiT on every frame: **the actions** (which keys / mouse moves)
and **the noise level** (how noisy that frame currently is). They are merged into one vector per
frame, called **c**, which adjusts every layer of the transformer (a mechanism called adaLN).

**What is NOT public:** Decart/Etched released the weights and an inference script, but **no
training code and no description of their training data or recipe**. Everything about training in
this document is the training pipeline *we* wrote (sections 6–9), reverse-engineered from their
inference code.

---

## 2. The earlier ideas Oasis is built from

Oasis is mostly a careful combination of known building blocks. Knowing them makes the rest easy.

| Building block (origin) | What it is | How Oasis uses it |
|---|---|---|
| **Vision Transformer, ViT** (Dosovitskiy et al. 2020) and **MAE-style** patch transformers (He et al. 2021) | Cut an image into square patches, turn each patch into a vector (a *token*), and run a transformer over the tokens | The **autoencoder** is a ViT: 20×20-pixel patches, 1024-wide tokens, encoder of 6 blocks, decoder of 12 blocks |
| **KL autoencoder / latent diffusion** (Rombach et al. 2022, "Stable Diffusion"; code style from VQGAN) | Train an autoencoder whose encoder outputs a *mean* and *log-variance* per latent value, then run diffusion in that small latent space instead of pixels | `DiagonalGaussianDistribution` in `vae.py`; Oasis always uses **only the mean** (no sampling) as the frame's code |
| **DiT, Diffusion Transformer** (Peebles & Xie 2023) | Replace the U-Net of diffusion models with a plain transformer over latent patches; inject the noise level with **adaLN-Zero** | The core of Oasis (`dit.py` cites DiT directly). Same patchify, same adaLN-Zero, same final layer, same zero-initialization |
| **Latte** (Ma et al. 2024) and other video transformers | Split attention into **spatial** (within a frame) and **temporal** (across frames) to make video affordable | Every Oasis block = spatial attention + MLP, then temporal attention + MLP ("axial" attention) |
| **Diffusion Forcing** (Chen et al. 2024) | Train a sequence model where **every frame gets its own independent noise level**, so at test time you can keep the past clean and the future noisy | Oasis's training objective and its sampling loop (both cite the Diffusion Forcing repo). Our training picks a separate random noise level for each of the 32 frames |
| **RoPE, rotary position embeddings** (Su et al. 2021), via the `rotary_embedding_torch` library | Encode *where* a token is by rotating its query/key vectors by position-dependent angles, instead of adding a position vector | **2-D axial RoPE** for spatial attention (row and column positions), **1-D RoPE** for temporal attention (frame index), and 2-D RoPE inside the autoencoder |
| **Sinusoidal timestep embedding** (Transformer / GLIDE / DiT) | Turn a single number (the noise level) into a vector of sines and cosines at many frequencies, then pass it through a small MLP | `TimestepEmbedder`: noise level → 256 sin/cos values → MLP → 1024-vector |
| **v-prediction** (Salimans & Ho 2022) | Instead of predicting the noise ε or the clean image x₀, predict a mix of both, **v = √ᾱ·ε − √(1−ᾱ)·x₀**; numerically stable at all noise levels | The DiT's output is v. Our loss is the MSE on v |
| **Sigmoid noise schedule** (Jabri et al. 2022, arXiv 2212.11972) | A recipe for how much signal survives at each of the 1,000 noise levels, shaped like a sigmoid | `sigmoid_beta_schedule(1000)` in `utils.py`, used identically in training and inference |
| **DDIM sampling** (Song et al. 2020) | A fast, deterministic way to go from noise to an image in a few big steps (here 10) instead of 1,000 | `generate.py`'s sampling loop (10 steps) and our validation/rollout scripts |
| **VPT action space** (OpenAI Video Pre-Training, Baker et al. 2022) | A standard 25-slot vector for Minecraft keyboard + mouse | Oasis's action input: 25 numbers per frame (`ACTION_KEYS` in `utils.py`) |
| **timm `Mlp`**, **PyTorch `scaled_dot_product_attention`** | Standard MLP block (Linear → GELU → Linear) and fused attention kernel | Every MLP and every attention call in both networks |
| **Hugging Face Accelerate** (our training) | Wraps PyTorch for multi-GPU data parallelism (DDP) and mixed precision | Our 4-GPU training in bf16 |

---

## 3. Network 1: the autoencoder (ViT-L/20, 229.2M weights, always frozen)

Code: `baselines/open-oasis/vae.py`, function `ViT_L_20_Shallow_Encoder`.
Config: patch size 20, encoder width 1024 × depth 6 × 16 heads, decoder width 1024 × depth 12 ×
16 heads, latent channels 16, input 360 × 640.
Weights: **encoder side ≈ 77M** (75.6M in its 6 blocks + 1.2M patch embedding + small heads),
**decoder side ≈ 152M** (151.2M in its 12 blocks + 1.2M output head). The decoder is twice as deep:
reconstructing a picture is harder than summarizing it.

### 3.1 Encoding one frame: the shape lifecycle

Follow a single frame. (In training, the batch and time axes are first merged, so the encoder sees
B·T = 3·32 = **96 independent frames** at once. More on that in section 8.)

1. **Raw frame:** `[3, 360, 640]`, uint8 pixels 0–255 from the video, converted to float and
   divided by 255, so values are in **[0, 1]**. 3 = red/green/blue, 360 rows, 640 columns.
2. **Rescale to [−1, 1]:** `x * 2 − 1`. Neural networks like inputs centered on zero.
3. **Patch embedding** (`PatchEmbed`, a `Conv2d(3 → 1024, kernel 20, stride 20)`): the frame is cut
   into non-overlapping **20 × 20-pixel squares**. 360 / 20 = **18 rows** of squares and 640 / 20 =
   **32 columns**, so **18 × 32 = 576 squares**. Each square's 20·20·3 = 1,200 pixel values are
   linearly projected into a **1024-number vector**: a **token**. Shape: `[576, 1024]` (576 tokens,
   each 1024 wide).
4. **6 encoder transformer blocks** (`AttentionBlock`), each doing:
   - `LayerNorm` → **multi-head self-attention** over all 576 tokens of this frame (16 heads of 64
     numbers; queries and keys rotated by **2-D axial RoPE** so each token knows its row and
     column) → output projection → **add back** to the input (residual connection);
   - `LayerNorm` → **MLP** (1024 → 4096 → GELU → 1024) → **add back**.
   Shape stays `[576, 1024]`: blocks refine the tokens, they never change their number or size.
5. **Final LayerNorm** (`enc_norm`): `[576, 1024]`.
6. **Bottleneck** (`quant_conv`, a `Linear(1024 → 32)`): each token is compressed to **32 numbers**:
   16 **means** and 16 **log-variances** (`DiagonalGaussianDistribution`). Oasis keeps **only the 16
   means** (`.mean`). Shape: `[576, 16]`.
7. **Scale by 0.07843137255** (Oasis's fixed `scaling_factor`). This rescales the codes so their
   spread suits the noise schedule (pure noise has standard deviation 1, and diffusion works best
   when the clean codes have a comparable scale).
8. **Reshape to a grid:** the 576 tokens go back to their 18 × 32 positions and the 16 numbers
   become **channels**: `[16, 18, 32]`. This is **the latent frame**, called **x₀** later.

**What `[16, 18, 32]` means:** an 18-row × 32-column map of the frame, where each cell summarizes
one 20×20-pixel square with 16 learned numbers (not colors: abstract features such as "grassy
texture", "sky", "edge going left"). 691,200 numbers → 9,216 numbers: **75× smaller**.

### 3.2 Decoding: the mirror image

Used only to turn generated codes back into pictures (inference, validation):
`[16,18,32]` → divide by 0.07843 → flatten to `[576, 16]` → `post_quant_conv` Linear(16 → 1024) →
**12 decoder blocks** (same structure as encoder blocks) → `dec_norm` → `predictor`
Linear(1024 → 1200 = 3·20·20) → **unpatchify** (put each 1,200-number token back as a 20×20×3 square
at its grid position) → `[3, 360, 640]` in [−1, 1] → `(x + 1) / 2` → clamp to [0, 1] → × 255 →
video frame.

---

## 4. Network 2: the Diffusion Transformer (DiT, 607.9M weights)

Code: `baselines/open-oasis/dit.py`, `DiT_S_2()` (the name is historical; it is not "small"):
**depth 16 blocks, width D = 1024, 16 heads × 64, MLP width 4096, patch size 2, 16 input/output
channels, max_frames = 32.**

### 4.1 Its three inputs (shapes for our batch B = 3, T = 32)

| Input | Shape | Meaning |
|---|---|---|
| **x_t**, the noisy latent clip | `[3, 32, 16, 18, 32]` | 3 clips × 32 frames × the 16×18×32 latent frame, each frame mixed with its own amount of noise |
| **t**, the noise levels | `[3, 32]` | one integer 0–999 **per frame** (0 = clean, 999 = almost pure noise) |
| **a**, the actions | `[3, 32, 25]` | one 25-slot action vector per frame: the keys/mouse that led **into** that frame |

### 4.2 Turning the inputs into tokens and a conditioning vector

**(a) Patchify the latents** (`x_embedder`, a `Conv2d(16 → 1024, kernel 2, stride 2)`). Batch and time
are merged first, `[3,32,16,18,32]` → `[96, 16, 18, 32]`, so every frame is handled independently.
The 18×32 latent grid is cut into **2 × 2 blocks of latent cells**: 18/2 = **9 rows** and 32/2 = **16
columns**, so **9 × 16 = 144 blocks per frame**. Each block (2·2·16 = 64 numbers) is projected to a
**1024-number token**. One token therefore covers 2×2 latent cells = **40 × 40 pixels** of the real
frame. Reshaped back with time: **`[3, 32, 9, 16, 1024]`** = clip × frame × token-row × token-column ×
token vector. Per clip: 32 × 144 = **4,608 tokens**.

Note there is **no added position vector** here: position comes entirely from RoPE inside attention.

**(b) Embed the noise level** (`TimestepEmbedder`). Each of the 3·32 = 96 integers t is turned into
**256 numbers** (128 cosines + 128 sines of t at geometrically spaced frequencies), then an MLP:
Linear(256 → 1024) → SiLU → Linear(1024 → 1024). Result: `[3, 32, 1024]`, one vector per frame.

**(c) Embed the actions** (`external_cond`; in our model a `GatedActionEmbed`): Linear(25 → 1024),
multiplied by a single learnable scalar **gate**. Result: `[3, 32, 1024]`.

**(d) Add them:** **c = t-embedding + action-embedding**, shape **`[3, 32, 1024]`**. This is the
**conditioning vector**: exactly one 1024-vector per frame. Every one of that frame's 144 tokens is
steered by the same c (the `modulate`/`gate` helpers broadcast it over the 9×16 grid).

### 4.3 The 16 SpatioTemporal blocks: the heart of Oasis

The token tensor `x` of shape `[3, 32, 9, 16, 1024]` passes through **16 identical blocks**
(`SpatioTemporalDiTBlock`). The shape never changes between blocks. Each block has **two halves**:
a **spatial half** (tokens of the same frame talk to each other) then a **temporal half** (the
same spot talks across frames). Each half has an attention step and an MLP step, and every step is
wrapped by **adaLN-Zero**.

**adaLN-Zero, the steering mechanism.** Each half owns a small layer
`adaLN_modulation = SiLU → Linear(1024 → 6·1024)`. Fed with c, it outputs **6 vectors of 1024**:
`shift₁, scale₁, gate₁` (for the attention step) and `shift₂, scale₂, gate₂` (for the MLP step).
Every step then does exactly:

```
h   = LayerNorm(x)                      # no learnable affine: the shift/scale come from c instead
h   = h * (1 + scale) + shift           # "modulate": c decides how to stretch/move each feature
h   = Step(h)                           # attention or MLP
x   = x + gate * h                      # residual, with c deciding how much of the update to keep
```

These modulation layers are **initialized to zero**, so at the start of (pre)training every block
is the identity function (`gate = 0`, so nothing is added). That's the "-Zero" in adaLN-Zero, and
it makes very deep models train stably.

**Spatial half (one frame at a time).**
1. `to_qkv`: Linear(1024 → 3·1024), no bias → queries, keys, values, each split into **16 heads × 64**.
2. Rearrange to **`[(3·32), 16, 9, 16, 64]`**: the 96 frames become 96 separate "sentences".
3. **2-D axial RoPE**: half of each head's 64 numbers is rotated by the token's **row**, the other
   half by its **column** (`freqs_for="pixel"`, max_freq 256). Now attention knows "this token is
   at row 3, column 7".
4. Flatten to `[96, 16, 144, 64]` and run **full (non-causal) attention** among the **144 tokens of
   that frame**: every patch looks at every other patch of the same frame.
5. Back to `[3, 32, 9, 16, 1024]` → `to_out` Linear(1024 → 1024).
6. Then the spatial **MLP**: Linear(1024 → 4096) → GELU (tanh approximation) → Linear(4096 → 1024).

*What it learns:* the layout and content of one picture (where the ground is, where trees are).

**Temporal half (one spatial location at a time).**
1. `to_qkv` again, but rearranged to **`[(3·9·16), 16, 32, 64]`**: every one of the 3·144 = 432
   token positions becomes a "sentence" of length **32**, i.e. the history of that 40×40-pixel spot
   across the 32 frames.
2. **1-D RoPE** on the frame index (0…31).
3. **Causal attention** (`is_causal=True`): frame k may only look at frames 0…k, never the future.
4. Back to `[3, 32, 9, 16, 1024]` → `to_out` → temporal **MLP** (same 1024 → 4096 → 1024).

*What it learns:* motion and consistency over time (how this spot changes when you walk forward or
turn the camera). Causality is what lets the model generate one frame after another.

**Why split attention this way ("axial attention")?** A clip has 4,608 tokens. Full attention
would compare 4,608² ≈ **21.2 million pairs** per head per layer. Spatial + temporal compares
144²·32 + 32²·144 ≈ **0.81 million pairs**, about **26× less**, at the price that information
needs two steps (space, then time) to travel diagonally.

**Weights per block:** each half has `to_qkv` 3.15M + `to_out` 1.05M + MLP 8.39M + adaLN 6.30M =
**18.9M**, so one block ≈ **37.8M**, and 16 blocks = **604.4M** (99.4% of the DiT). Almost half of
everything sits in the MLPs.

### 4.4 The final layer: back to the latent shape

`FinalLayer`: LayerNorm → modulate with its own adaLN (`shift, scale` from c, 2 × 1024) →
Linear(1024 → **64**). 64 = 2·2·16: each token predicts the 16 channels for the **2 × 2 latent cells**
it came from. **Unpatchify** puts those back on the grid: `[96, 9, 16, 64]` → `[96, 16, 18, 32]` →
**`[3, 32, 16, 18, 32]`**. This output is **v̂** ("v-hat"), the model's prediction of the velocity
v for every frame. It has exactly the same shape as the input x_t.

### 4.5 Where the 607.9M weights live

| Part | Weights |
|---|---|
| 16 SpatioTemporal blocks | 604.37M |
| Final layer (adaLN 2.10M + Linear 0.07M) | 2.16M |
| Timestep embedder MLP | 1.31M |
| Patch embedding (Conv2d 16 → 1024, k=2) | 0.07M |
| Action layer (Linear 25 → 1024 + gate) | 0.03M (26,625 numbers) |
| **Total** | **607.9M** |

---

## 5. The diffusion math, slowly

**Noise levels.** There are 1,000 levels, t = 0 … 999. A fixed table **ᾱ_t** ("alpha-bar") says how
much of the clean signal survives at level t. It comes from the **sigmoid schedule**
(`sigmoid_beta_schedule(1000)`): per-step noise amounts β_t, then ᾱ_t = (1−β₀)(1−β₁)…(1−β_t). ᾱ is
close to 1 at t = 0 (clean) and close to 0 at t = 999 (all noise), falling in an S-shape.

**Adding noise (forward process).** For a clean latent frame x₀ and Gaussian noise ε of the same
shape:

```
x_t = sqrt(ᾱ_t) · x₀ + sqrt(1 − ᾱ_t) · ε
```

At t near 0 you mostly see x₀; at t near 999 you mostly see ε.

**What the network predicts (v-prediction).** The target is the **velocity**

```
v = sqrt(ᾱ_t) · ε − sqrt(1 − ᾱ_t) · x₀
```

Think of x_t as a point on a circle between "clean" (x₀) and "noise" (ε): v is the direction you
move along that circle as the noise level changes. Predicting v is well-behaved at every noise
level: predicting ε is hard when the frame is almost clean, and predicting x₀ is hard when it is
almost pure noise.

**Getting x₀ and ε back from a prediction v̂** (used at inference):

```
x̂₀ = sqrt(ᾱ_t) · x_t − sqrt(1 − ᾱ_t) · v̂
ε̂  = ( sqrt(1/ᾱ_t) · x_t − x̂₀ ) / sqrt(1/ᾱ_t − 1)
```

---

## 6. Our data pipeline: from Luanti episodes to training batches

Code: `baselines/finetune_oasis_data.py` (`CraftiumOasisDataset`) and
`baselines/craftium_action_features.py`.

### 6.1 What one episode contains
Each episode of the `percool/luanti` hudnav data (1,000 episodes used for training) is a folder with:
- `rgb.mp4`: **600 frames** of 360 × 640 gameplay video;
- `data.npz`: per-frame game data, of which we use `action` **[600, 23]** (Luanti's 23 control
  flags), `player_yaw` and `player_pitch` **[600]** (where the camera points, in degrees).

### 6.2 Cutting episodes into training windows ("sliding" mode)
A training sample is a window of **32 consecutive frames**. Every possible start position is used:
starts 0, 1, 2, …, 568 (since 568 + 32 = 600), so **569 windows per episode × 1,000 episodes =
569,000 windows**. They are pooled into one big list and shuffled. Only the 32 needed frames are
decoded (with PyAV), not the whole video.

Pixels: uint8 → float ÷ 255 → **[32, 3, 360, 640] in [0, 1]** (frame × color × row × column).

### 6.3 Translating Luanti's controls into Oasis's 25 slots
Oasis was trained with Minecraft's 25-slot VPT action vector (`ACTION_KEYS` in `utils.py`):
`inventory, ESC, hotbar.1…hotbar.9, forward, back, left, right, cameraX, cameraY, jump, sneak,
sprint, swapHands, attack, use, pickItem, drop`. We rebuild exactly that vector from Luanti's data
(`augment_actions_to_oasis_native_layout`), so the pretrained action weights keep their meaning:

| Oasis slot | Filled from Luanti |
|---|---|
| hotbar.1–9 | slot_1–9 |
| forward, back, left, right | forward, backward, left, right |
| jump, sneak, sprint | jump, sneak, aux1 |
| attack, use | dig (left click), place (right click) |
| drop | drop |
| cameraX, cameraY | real yaw / pitch change since the previous frame (only when that mouse axis was active), converted with Oasis's own formula: degrees ÷ 0.5 ÷ 40, clipped to [−1, 1] |
| inventory, ESC, swapHands, pickItem | no Luanti equivalent → always 0 |

So **21 of 25 slots carry information**. Shape per episode: `[600, 25]`.

### 6.4 Lining actions up with frames (the "shift by one")
Oasis's convention (from `load_actions`): **the action stored at frame t is the action that
produced the transition into frame t.** So the window's actions are shifted by one: action[0] is
the action *before* the window's first frame (the real previous action, or all zeros if the window
starts at the very beginning of the episode), action[1] is the action taken at frame 0, and so on.
Shape per window: **`[32, 25]`**.

### 6.5 Batches
The PyTorch `DataLoader` stacks 3 windows per GPU (`batch_size=3`, shuffled, last incomplete batch
dropped). With 4 GPUs, **12 clips (= 384 frames) per optimizer step**:

| Tensor (per GPU) | Shape | Meaning |
|---|---|---|
| frames | `[3, 32, 3, 360, 640]` | clip × frame × RGB × row × column, values in [0, 1] |
| actions | `[3, 32, 25]` | clip × frame × action slot |

Two small fixed sets are also built for monitoring, each with **one window per episode, always
frames 0–31**, so scores are comparable across steps: **val** (25 held-out episodes, never trained
on) and **train_eval** (25 training episodes).

---

## 7. Model surgery before training starts

Code: `load_oasis_for_finetuning` in `finetune_oasis.py`.

1. Build the DiT with a **new action layer** of input size 25, and load all pretrained Oasis weights
   **except** the old action layer (`external_cond.*`).
2. **Informed initialization:** copy the pretrained action weights for the 21 slots we fill (and the
   bias) into the new layer, so "forward" means forward from step one, instead of starting from zero.
3. **Wrap it in `GatedActionEmbed`:** output = Linear(action) × gate, with the **gate starting at 1**.
   (An earlier version started both the gate and the weights at 0. Since the output is
   gate × weights, the gradient of each is proportional to the other, so both gradients were
   exactly 0 forever and the action layer never learned. Starting the gate at 1 fixed it.)
4. Load the autoencoder, set it to eval mode and **freeze it** (`requires_grad=False`).
5. Save this untouched starting point as `step_000000.pt`: the honest "before fine-tuning" model.

---

## 8. One training step, end to end

Code: the training loop in `finetune_oasis.py` (around lines 690–730). Shapes are per GPU.

```
frames [3,32,3,360,640]   actions [3,32,25]
        │                        │
        ▼                        │
 merge batch & time  [96,3,360,640]
        │                        │
 ×2−1, ViT-L ENCODER (frozen, no gradients, bf16)
        │                        │
 mean × 0.07843  → [96,576,16] → reshape → x₀ [3,32,16,18,32]
        │                        │
        │   t ~ Uniform{0..999}, shape [3,32]  ← independent per frame (Diffusion Forcing)
        │   ε ~ N(0,1),  shape [3,32,16,18,32]
        ▼                        │
 x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε      │        target v = √ᾱ_t·ε − √(1−ᾱ_t)·x₀
        │                        │
        ▼                        ▼
   DiT(x_t, t, actions)  (bf16 autocast)  →  v̂ [3,32,16,18,32]
        │
 loss L = mean over all 3·32·16·18·32 = 884,736 numbers of (v̂ − v)²
        │
 backward → DDP averages gradients over the 4 GPUs → clip gradient norm to 1.0
        │
 AdamW step (two learning rates) → LR scheduler step → next batch
```

Step by step:

1. **Load a batch:** frames `[3,32,3,360,640]`, actions `[3,32,25]`.
2. **Encode (no gradients):** merge batch and time to `[96,3,360,640]`, run the frozen encoder in
   bf16, keep the mean × 0.07843 → `[96,576,16]` → reshape → **x₀ `[3,32,16,18,32]`**.
3. **Pick a noise level per frame:** `t = randint(0, 1000)` with shape `[3,32]`. Every frame of
   every clip gets its own level. This is **Diffusion Forcing**, and it is the single most
   important design choice: the model constantly sees clips where some frames are clean and others
   are noisy, which is exactly the situation at play time (clean past, noisy next frame).
4. **Noise it:** ε ~ N(0,1) with x₀'s shape; **x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε**, where ᾱ_t is looked up
   per frame and broadcast over that frame's 16×18×32 values.
5. **Target:** **v = √ᾱ_t·ε − √(1−ᾱ_t)·x₀**.
6. **Forward:** **v̂ = DiT(x_t, t, actions)** in bf16 mixed precision (section 4).
7. **Loss:** mean squared error, **L = mean((v̂ − v)²)**, computed in fp32 over all 884,736 numbers
   (every frame, channel and position counts equally).
8. **Backward and sync:** gradients are computed on each GPU; DDP (via Accelerate) averages them
   across the 4 GPUs, so the effective batch is **12 clips**.
9. **Clip** the total gradient norm to **1.0** (prevents rare huge updates).
10. **AdamW update** with **two parameter groups**: the 607.9M pretrained weights at **lr 1×10⁻⁶**
    (change Oasis very gently so it keeps its drawing skills), the new action layer at **lr 1×10⁻⁴**
    (100× faster, it has the most to learn). Other AdamW settings are PyTorch defaults: betas
    (0.9, 0.999), eps 1e-8, weight decay 0.01.
11. **Learning-rate schedule:** linear **warm-up over 200 steps** (factor (step+1)/200), then constant.
12. **Every 100 steps:** validation (section 9). **Every 500 steps:** save model + optimizer +
    scheduler. Our run went to **61,500 steps**; the 40,000 checkpoint was used for the final scores.

---

## 9. Validation during training (every 100 steps)

Code: `compute_eval_metrics` in `finetune_oasis.py`. It runs on GPU 0 only, on the fixed `val` and
`train_eval` windows (frames 0–31, 25 episodes each).

1. **Fixed-level loss:** the same v-prediction MSE, but with **t = 500 for every frame** (instead of
   random), so the number is directly comparable from one check to the next.
2. **One-step-ahead generation:** the first 31 frames are real (encoded from pixels) and kept at a
   low, fixed noise level (**14**, Oasis's "stabilization level" 15 minus one); the **32nd frame**
   starts as pure noise and is generated with **10 DDIM steps** (section 10). It is decoded and
   compared with the real 32nd frame using **PSNR**, **SSIM** and **LPIPS** (AlexNet).

This measures "can it draw the next frame given a perfect past". It does **not** measure long-video
drift: for that we ran separate 200/400/600-frame rollouts scored with FVD and FID.

---

## 10. Inference: how Oasis "plays" frame after frame

Code: `generate.py` (and the same logic in our rollout/eval scripts).

1. **Start:** encode the prompt frame(s) to latents, e.g. `[1, 1, 16, 18, 32]` for one frame.
2. **For each new frame i:**
   - append a fresh noise frame (Gaussian, clamped to ±20): x becomes `[1, i+1, 16, 18, 32]`;
   - keep only the **last 32 frames** (the sliding window, because the model was trained on 32);
   - **context frames** are labeled with noise level **14** ("almost clean"); the **new frame**
     starts at level **999**;
   - **10 DDIM steps** over the levels 999 → 899 → … → 99 → 0
     (`noise_range = linspace(−1, 999, 11)`). At each step: run the DiT, get v̂, compute x̂₀ and ε̂
     (section 5), and move the **new frame only** to the next lower level:
     x ← √ᾱ_next·x̂₀ + √(1−ᾱ_next)·ε̂. Context frames are never changed. At the last step the frame
     is set to x̂₀ directly;
   - the frame's actions are fed in alongside the window.
3. **Decode** every latent frame with the ViT-L decoder (÷ 0.07843 → decoder → (x+1)/2 → clamp →
   × 255) and write the video.

**Why training with per-frame noise matters here:** at play time, the model always sees "nearly
clean past + very noisy present". Diffusion Forcing training showed it exactly such mixtures. The
small noise level (14) on context frames also tells the model "these past frames may contain small
mistakes", which reduces (but does not remove) **drift**: each new frame is built on the model's
own earlier frames, so small errors compound over hundreds of frames (exposure bias). This is the
failure mode visible in our 600-frame results.

---

## 11. Drawing instructions (give these to an LLM or a designer)

Use rounded boxes for modules, plain arrows for data, write the tensor shape **on each arrow**,
grey boxes for **frozen** modules, a dark box for the **trainable** DiT, and an orange box for the
**new** action layer. Batch example: B = 3, T = 32.

**D1. Overall architecture (left to right).**
- Box "Video clip" `[B,T,3,360,640]` → arrow → grey box "ViT-L/20 encoder (frozen, 77M):
  20×20 patches, 6 blocks" → arrow labeled `mean × 0.0784` → box "Latents x₀" `[B,T,16,18,32]` →
  arrow → dashed box "Add noise, own level t per frame" → arrow `x_t [B,T,16,18,32]` → dark box
  "Diffusion Transformer (608M): Patchify 2×2 → 16 SpatioTemporal blocks → Final layer" → arrow →
  box "v̂" `[B,T,16,18,32]`.
- Below the DiT: box "Actions `[B,T,25]`" → orange box "Gated linear 25→1024" ; box "Noise level
  t `[B,T]`" → box "sin/cos 256 → MLP → 1024" ; both arrows meet at a ⊕ labeled
  "c `[B,T,1024]`", and an arrow goes **up into every block** of the DiT, labeled "adaLN".
- To the right of v̂: a dashed path "inference only" → grey box "ViT-L/20 decoder (frozen, 152M):
  12 blocks" → "Video `[B,T,3,360,640]`".

**D2. One SpatioTemporal DiT block (in the style of the original DiT paper's figure).**
- Input "Tokens `[B,T,9,16,1024]`" at the bottom, "Condition c `[B,T,1024]`" bottom right feeding
  an "MLP (SiLU → Linear 1024→6144)" that fans out arrows labeled γ, β, α.
- Lower stack, labeled **"Spatial half: attention within each frame (144 tokens, 2-D RoPE)"**:
  LayerNorm → "Scale, Shift (γ₁, β₁)" → "Multi-head self-attention, 16 heads × 64" →
  "Scale (α₁)" → ⊕ (residual from input) → LayerNorm → "Scale, Shift (γ₂, β₂)" →
  "MLP 1024→4096→1024 (GELU)" → "Scale (α₂)" → ⊕.
- Upper stack, same shape, labeled **"Temporal half: causal attention across the 32 frames for
  each position (1-D RoPE)"**, with its own adaLN MLP and γ₃, β₃, α₃, γ₄, β₄, α₄.
- Side note "× 16 blocks". A small inset can show the two attention masks: a full 144×144 square
  (spatial) and a lower-triangular 32×32 square (temporal, causal).

**D3. Training step pipeline.** Draw the ASCII diagram of section 8 as boxes: Dataset (sliding
32-frame windows) → frozen encoder → x₀ → (t per frame, ε) → x_t and target v → DiT (with actions
and t) → v̂ → "MSE(v̂, v)" → backward → "DDP all-reduce (4 GPUs)" → "clip 1.0" → "AdamW: 1e-6 /
1e-4, warm-up 200". Mark the encoder with a snowflake (frozen) and the DiT with a flame (trained).

**D4. Inference loop.** A loop arrow around: "window of last ≤32 latent frames (context at
level 14) + 1 new noise frame (level 999)" → "DiT → v̂ → x̂₀, ε̂ → DDIM update (new frame only)" ×10
steps → "append finished frame" → back to the start; on the side, "decode with ViT-L decoder → video".

---

## 12. Cheat sheet

### 12.1 Every shape in one table (B = 3, T = 32)

| Where | Shape | Meaning |
|---|---|---|
| Raw window | `[3, 32, 3, 360, 640]` | clip × frame × RGB × row × column, [0,1] |
| Encoder input (merged) | `[96, 3, 360, 640]` | 96 independent frames, [−1,1] |
| Encoder tokens | `[96, 576, 1024]` | 18×32 = 576 patches of 20×20 px, 1024 numbers each |
| Bottleneck | `[96, 576, 32]` → mean `[96, 576, 16]` | 16 means + 16 log-variances; keep means |
| Latent clip x₀ / noisy x_t / target v / prediction v̂ | `[3, 32, 16, 18, 32]` | clip × frame × channel × latent row × latent column |
| Noise levels t | `[3, 32]` | one level 0–999 per frame |
| Actions | `[3, 32, 25]` | one 25-slot vector per frame |
| DiT tokens | `[3, 32, 9, 16, 1024]` | 144 tokens per frame (2×2 latent cells = 40×40 px each) |
| Conditioning c | `[3, 32, 1024]` | t-embedding + action embedding, one per frame |
| Spatial attention view | `[96, 16, 144, 64]` | frames × heads × tokens of a frame × head width |
| Temporal attention view | `[432, 16, 32, 64]` | positions (3·144) × heads × frames × head width |
| Final layer output | `[3, 32, 9, 16, 64]` → unpatchify | 64 = 2·2·16 values per token |

### 12.2 Every hyperparameter

| Item | Value |
|---|---|
| Autoencoder | ViT-L/20: patch 20, width 1024, encoder 6 blocks, decoder 12 blocks, 16 heads, latent 16 channels, 229.2M weights, frozen; scale factor 0.07843137255 |
| DiT | depth 16, width 1024, 16 heads × 64, MLP 4096, patch 2, max 32 frames, 607.9M weights |
| Noise | 1,000 levels, sigmoid β schedule, v-prediction, per-frame random level (Diffusion Forcing) |
| Loss | MSE between v̂ and v over all frames and values |
| Data | 1,000 Luanti hudnav episodes × 600 frames; 32-frame sliding windows (569,000) |
| Batch | 3 clips per GPU × 4 A100 GPUs = 12 clips (384 frames) per update |
| Optimizer | AdamW, lr 1e-6 (pretrained weights) / 1e-4 (action layer), betas (0.9, 0.999), weight decay 0.01, grad-norm clip 1.0 |
| Schedule | linear warm-up 200 steps, then constant |
| Precision | bf16 mixed precision (encoder and DiT), loss in fp32 |
| Length | 61,500 steps; 40,000-step checkpoint used for final scores |
| Validation | every 100 steps: loss at t=500 and one-step-ahead PSNR/SSIM/LPIPS (10 DDIM steps), 25 val + 25 train episodes |
| Inference | 32-frame sliding window, context at noise level 14, 10 DDIM steps per new frame |

### 12.3 Glossary

- **Token:** a vector of numbers (here 1024) representing one small region of an image.
- **Latent:** the compressed code of a frame produced by the encoder (`[16,18,32]`).
- **Channel:** one of the numbers describing a single grid position (3 for RGB, 16 for latents).
- **Attention:** each token builds a query, key and value; it mixes in the values of the tokens
  whose keys best match its query. **Multi-head** = 16 independent attentions of width 64.
- **Causal attention:** a token may only attend to earlier (or equal) time steps.
- **RoPE:** position information injected by rotating queries and keys; 2-D for rows/columns,
  1-D for frame index.
- **LayerNorm:** rescales each token to zero mean and unit variance.
- **adaLN(-Zero):** LayerNorm whose scale, shift and output gate are computed from the
  conditioning c; "Zero" = those layers start at zero so each block starts as the identity.
- **MLP:** per-token feed-forward network (1024 → 4096 → GELU → 1024).
- **Residual connection:** a block's output is added to its input instead of replacing it.
- **Diffusion model:** learns to remove noise; generating = starting from noise and denoising.
- **v-prediction:** the network predicts v = √ᾱ·ε − √(1−ᾱ)·x₀ rather than ε or x₀.
- **Diffusion Forcing:** an independent noise level per frame during training.
- **DDIM:** a deterministic sampler that denoises in a few big steps.
- **Exposure bias / drift:** errors pile up because, at play time, the model conditions on its own
  imperfect outputs instead of real frames.
- **DDP:** distributed data parallel: each GPU processes its own mini-batch and gradients are
  averaged across GPUs before every update.
- **bf16:** a 16-bit number format that halves memory and speeds up math with little precision loss.
