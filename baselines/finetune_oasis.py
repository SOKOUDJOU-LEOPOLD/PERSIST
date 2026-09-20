"""Fine-tune Oasis's DiT on Craftium data, swapping its 25-dim VPT-style action-conditioning
layer for a 23-dim layer matching Craftium's native action space (see the plan: this repo's own
`dit.py:external_cond` is a single clean `nn.Linear`, isolated enough to swap without touching
anything else).

Training objective, reverse-engineered from generate.py's sampling loop (no training code shipped
with open-oasis -- see the earlier finding that this repo has zero training code at all):
  - Noise schedule: `sigmoid_beta_schedule(1000)` (utils.py), same as inference.
  - v-prediction parameterization: generate.py computes
      x_start = sqrt(alphas_cumprod[t]) * x_t - sqrt(1 - alphas_cumprod[t]) * v
    from the model's output `v`, which is exactly the standard v-prediction definition
    (Salimans & Ho 2022): v = sqrt(alphas_cumprod[t]) * noise - sqrt(1 - alphas_cumprod[t]) * x_start.
    So the model is trained to predict that same `v` given a noised x_t.
  - "Diffusion Forcing" (the paper this repo's docstring cites): each frame in a training window
    gets an INDEPENDENT, uniformly-random noise level, not one shared timestep for the whole
    clip -- this is what lets the model condition on a mix of clean/near-clean context frames and
    a noisy target frame at inference time (generate.py's `stabilization_level` context frames vs.
    the newly-appended noisy frame). Training replicates that by sampling t independently per
    frame per example.

Example:
    cd baselines/open-oasis
    python ../finetune_oasis.py --smoke-test-steps 50
"""
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from finetune_oasis_data import CraftiumOasisDataset  # noqa: E402
from craftium_action_features import (  # noqa: E402
    CRAFTIUM_ACTION_DIM_OASIS_NATIVE,
    OASIS_NATIVE_INFORMED_INIT_ACTION_MAP,
)

CRAFTIUM_ACTION_DIM = 23


class GatedActionEmbed(torch.nn.Module):
    """Drop-in replacement for DiT's plain `nn.Linear(action_dim, hidden_size)` external_cond
    layer: same call signature (`self(action) -> (..., hidden_size)`), but the output is scaled
    by a learnable scalar gate before being returned. Since dit.py's forward does
    `c += self.external_cond(external_cond)` unconditionally, this needs no changes to dit.py at
    all -- it's substituted in after construction (see load_oasis_for_finetuning).

    Motivation: even with a zero-initialized linear layer, once trained its raw output is added
    into `c` completely unscaled -- nothing stops training from growing that contribution to a
    magnitude that measurably shifts the shared conditioning signal for EVERY frame, degrading
    base visual fidelity even while the action-following objective improves (this is the
    hypothesized cause of the color/lighting drift seen in every fine-tuning attempt so far). A
    learnable gate lets the model control the action signal's magnitude explicitly and gradually.

    BUG FIX (found by directly loading trained checkpoints and confirming, empirically, that
    `external_cond`'s output was byte-identical for real/random/zero actions after thousands of
    training steps -- i.e. the action-conditioning pathway had never received a single nonzero
    gradient, in every fine-tuning run so far): the gate previously started at exactly 0, matching
    `linear`'s own zero-initialized weight/bias. Output = `gate * (W @ action + b)` is a PRODUCT of
    two independently zero-initialized learnable quantities, and by the product rule, the gradient
    w.r.t. either factor is proportional to the OTHER factor -- so with both at exactly 0, every
    gradient (`d(gate)`, `d(W)`, `d(b)`) is exactly 0, forever; this is a permanent fixed point
    under gradient descent, not something more training steps can escape. The referenced
    `zero_init_gate` option in `baselines/WorldMem/experiments/exp_base.py` does NOT have this
    problem -- it zeros a slice of a weight matrix whose *input* (a conditioning embedding) is
    never itself zero, so that gate's gradient is nonzero from step 1. This class's gate, by
    contrast, multiplied a branch whose input (`linear`'s raw output) was ALSO forced to zero --
    the two zero-inits were never safe to combine. Fix: initialize `gate` to a nonzero value while
    keeping `linear` zero-initialized. Output at step 0 is still exactly 0 (since `linear`'s output
    is exactly 0, `0 * gate == 0` for any `gate`), so the "no initial perturbation to the pretrained
    model" property is preserved -- but now `d(loss)/d(linear.weight) = gate * action != 0`, so
    `linear` can start learning from the very first gradient step, and once it does, `gate`'s own
    gradient (`d(loss)/d(gate) = linear(action)`) becomes nonzero too."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(action_dim, hidden_size)
        self.gate = torch.nn.Parameter(torch.ones(1))

    def forward(self, action):
        return self.linear(action) * self.gate


def build_dit_architecture(gated_action: bool = True, zero_init_action_layer: bool = True, action_dim: int = CRAFTIUM_ACTION_DIM):
    """Construct the action DiT architecture only (no checkpoint weights loaded) -- shared by
    load_oasis_for_finetuning (training) and every generation/eval script that needs to load one
    of our own fine-tuned checkpoints, so the architecture (in particular, whether external_cond
    is a plain nn.Linear or a GatedActionEmbed, and its action_dim) always matches whatever the
    checkpoint being loaded was actually saved with."""
    from dit import DiT

    model = DiT(patch_size=2, hidden_size=1024, depth=16, num_heads=16, external_cond_dim=action_dim)
    if zero_init_action_layer:
        torch.nn.init.zeros_(model.external_cond.weight)
        torch.nn.init.zeros_(model.external_cond.bias)
    if gated_action:
        gated = GatedActionEmbed(action_dim, model.external_cond.out_features)
        gated.linear.weight.data.copy_(model.external_cond.weight.data)
        gated.linear.bias.data.copy_(model.external_cond.bias.data)
        model.external_cond = gated
    return model


def load_finetuned_dit(ckpt_path: str, device: str):
    """Load one of our own saved fine-tuned checkpoints, auto-detecting whether it was saved with
    a plain nn.Linear external_cond or a GatedActionEmbed-wrapped one, instead of assuming
    build_dit_architecture()'s current default.

    This matters concretely: our early "freeze-backbone-only" experiment (finetune_oasis_v4/*)
    was trained and saved BEFORE GatedActionEmbed existed at all (plain "external_cond.weight" /
    "external_cond.bias" keys), while the later "gated + adaLN" experiment
    (finetune_oasis_v5/*) was trained after (an "external_cond.gate" / "external_cond.linear.*"
    triplet). build_dit_architecture()'s default flipped to gated_action=True when
    GatedActionEmbed was introduced -- correct for v5, but that silently made every v4 checkpoint
    unloadable by any script that just calls build_dit_architecture() with no arguments. Detecting
    from the checkpoint's own keys, rather than hardcoding a default, is the fix that works for
    both checkpoint families (and any future one) without the caller needing to know or guess
    which era a given checkpoint came from."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    gated_action = "external_cond.gate" in state_dict
    # action_dim inferred directly from the saved weight's own shape (in_features), rather than
    # from the checkpoint's args dict, so this stays correct even for checkpoints saved before
    # any given action-layout flag existed at all.
    weight_key = "external_cond.linear.weight" if gated_action else "external_cond.weight"
    action_dim = state_dict[weight_key].shape[1]
    model = build_dit_architecture(gated_action=gated_action, action_dim=action_dim)
    model.load_state_dict(state_dict)
    return model.to(device).eval(), ckpt.get("step")


def load_oasis_for_finetuning(
    oasis_ckpt: str, vae_ckpt: str, device: str, zero_init_action_layer: bool = True, gated_action: bool = True,
    action_dim: int = CRAFTIUM_ACTION_DIM, informed_action_init: bool = False,
    informed_init_map: dict = None,
):
    """Load Oasis's DiT with a freshly-initialized action layer, everything else from the
    pretrained checkpoint. See module docstring / the plan's "model surgery" step.

    `zero_init_action_layer`: zero-initializes the new layer's weight AND bias (rather than
    nn.Linear's default random init), so at step 0 `external_cond(actions) == 0` for every
    action -- i.e. training starts from the literal pretrained model's unmodified behavior
    (c += 0) and has to learn to incorporate the action signal, rather than immediately injecting
    a random/systematically-biased offset into the shared conditioning vector `c`.

    `gated_action`: wraps the action layer in GatedActionEmbed (see above) instead of a plain
    nn.Linear -- adds an explicit, learnable, zero-initialized scalar controlling the action
    signal's overall magnitude, on top of zero_init_action_layer's per-weight zero-init.

    `informed_action_init`: instead of leaving the new action layer entirely zero-initialized,
    copy the pretrained 25-dim checkpoint's own weight COLUMNS for the dims in `informed_init_map`
    (source idx -> pretrained idx) into the corresponding new columns. Meaningful for
    `action_dim=CRAFTIUM_ACTION_DIM_OASIS_NATIVE` (native layout, `informed_init_map` defaults to
    `OASIS_NATIVE_INFORMED_INIT_ACTION_MAP`, an identity map since native-layout columns already
    coincide with Oasis's own) -- not meaningful for the plain 23-dim bespoke layout, which has no
    established mapping to Oasis's own column semantics at all."""
    from vae import VAE_models

    if informed_init_map is None:
        informed_init_map = OASIS_NATIVE_INFORMED_INIT_ACTION_MAP

    # Built WITHOUT gating first (gated_action=False here regardless of the caller's setting) so
    # informed_action_init below can act on a plain nn.Linear's .weight/.bias directly -- gating
    # (if requested) is applied manually afterward, copying whatever the plain layer ends up
    # holding (zero-init, or informed-init'd) into the GatedActionEmbed wrapper. Doing it in the
    # opposite order (gate first, informed-init second) would try to index .weight/.bias on
    # GatedActionEmbed itself, which has no such attributes (it has .linear.weight/.linear.bias
    # and .gate instead) -- mirrors finetune_worldmem.py's load_worldmem_for_finetuning ordering.
    model = build_dit_architecture(gated_action=False, zero_init_action_layer=zero_init_action_layer, action_dim=action_dim)
    pretrained = load_file(oasis_ckpt)
    mismatched = {k for k in pretrained if k.startswith("external_cond.")}
    filtered = {k: v for k, v in pretrained.items() if k not in mismatched}

    action_layer = model.external_cond
    if informed_action_init:
        pretrained_weight = pretrained.get("external_cond.weight")
        pretrained_bias = pretrained.get("external_cond.bias")
        if pretrained_weight is None or pretrained_weight.shape[1] != 25:
            raise ValueError(
                f"informed_action_init=True but couldn't find a 25-dim pretrained external_cond "
                f"weight in the checkpoint (got {None if pretrained_weight is None else pretrained_weight.shape})."
            )
        with torch.no_grad():
            for src_idx, oasis_idx in informed_init_map.items():
                action_layer.weight[:, src_idx].copy_(pretrained_weight[:, oasis_idx])
            action_layer.bias.copy_(pretrained_bias)
        print(f"informed_action_init=True: copied {len(informed_init_map)} pretrained action "
              f"columns {informed_init_map} into the new {action_dim}-dim layer; bias copied in full.")

    if gated_action:
        gated = GatedActionEmbed(action_dim, action_layer.out_features)
        gated.linear.weight.data.copy_(action_layer.weight.data)
        gated.linear.bias.data.copy_(action_layer.bias.data)
        model.external_cond = gated

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # Every key should be one of: the 2 intentionally-dropped external_cond.* tensors (now
    # correctly reinitialized at the new 23-dim shape by the fresh nn.Linear), or a
    # RotaryEmbedding `*.freqs` buffer (non-persistent -- recomputed deterministically at
    # __init__ from the module's own config, never saved in the checkpoint at all; confirmed by
    # running this and inspecting exactly which keys came back missing). Any OTHER missing/
    # unexpected key would mean the checkpoint didn't actually match this architecture, which
    # would silently produce garbage, so that case is still checked rather than ignored.
    # Note: when gated_action=True, model.external_cond's own parameter names are
    # external_cond.linear.{weight,bias}/external_cond.gate (GatedActionEmbed), not
    # external_cond.{weight,bias} -- matched by prefix here rather than exact name for that
    # reason (mismatched itself is still computed from the CHECKPOINT's own key names above,
    # which are always the plain nn.Linear naming regardless of gated_action).
    expected_missing = {k for k in missing if k.startswith("external_cond.")} | {
        k for k in missing if k.endswith("rotary_emb.freqs")
    }
    unexpected_missing = set(missing) - expected_missing
    assert not unexpected_missing, f"Unexpected missing keys (checkpoint/architecture mismatch): {unexpected_missing}"
    assert not unexpected, f"Unexpected extra keys in checkpoint: {unexpected}"
    model = model.to(device)

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    load_file_result = load_file(vae_ckpt)
    vae.load_state_dict(load_file_result)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return model, vae


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """Copied verbatim from baselines/open-oasis/utils.py -- kept identical so the noise schedule
    used for fine-tuning exactly matches the schedule the pretrained weights were trained under."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def psnr_per_sample(gen: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """gen, real: (B, 3, H, W) float in [0, 1]. Returns (B,) -- one PSNR per sample (same formula
    used everywhere else in this project, e.g. generate_oasis_native_grid.py's psnr(), just
    batched and vectorized here instead of a per-frame Python loop, and averaged the same way:
    per-sample first, then mean -- not a single pooled-MSE-over-the-whole-batch number, which
    would be a subtly different (if similar) quantity)."""
    mse = torch.mean(((gen * 255.0) - (real * 255.0)) ** 2, dim=[1, 2, 3])  # (B,)
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10(255.0**2 / mse)


@torch.no_grad()
def compute_eval_metrics(
    model, vae, loader, device, alphas_cumprod_flat: torch.Tensor, scaling_factor: float,
    ssim_metric: StructuralSimilarityIndexMeasure, lpips_metric: LearnedPerceptualImagePatchSimilarity,
    ddim_steps: int = 10, stabilization_level: int = 15,
) -> dict:
    """Runs the cheap "one-step-ahead" validation/train-eval check: for every (frames, actions)
    window in `loader`, ALL context frames [0..T-2] are real (VAE-encoded from real pixels, held
    at stabilization_level noise -- the same "trustworthy context" convention
    finetune_oasis_rollout.py's generate_rollout uses at inference time), and ONLY the single
    target frame [T-1] is genuinely sampled: starts from pure noise, run through `ddim_steps`
    reverse-diffusion steps exactly like generate_rollout's own inner loop (same math, copied
    here rather than reused directly -- generate_rollout is hard-wired for a single real seed
    frame plus fully-autoregressive-generated context, not "many real context frames, one sampled
    target," which is what this needs). No autoregressive chaining: every window's target
    prediction is conditioned on REAL frames only, so this measures one-step prediction quality in
    isolation, not compounding rollout drift -- deliberately a different, much cheaper metric than
    the full-rollout PSNR the post-training sweep (eval_oasis_checkpoint_sweep.py) computes.

    Also returns a plain v-prediction MSE loss at a fixed mid noise level (same formula training
    uses, but a FIXED t=500 for every frame instead of training's random-per-frame t, so repeated
    calls with the same window are exactly comparable across checkpoints).

    SSIM and LPIPS are computed on the same one-step-ahead generated-vs-real target frame pair as
    PSNR -- not a separate, more expensive pass -- via `ssim_metric`/`lpips_metric`, torchmetrics
    modules instantiated once in main() and reused (LPIPS in particular loads a pretrained AlexNet
    backbone, so it must not be recreated per call). LPIPS' `net_type='alex'` matches WorldMem's
    own eval code (algorithms/common/metrics/lpips.py) for direct cross-baseline comparability.

    model.eval() is the caller's responsibility (and switching back to model.train() afterward) --
    kept out of this function so it composes cleanly whether called for val or train-eval.

    Returns {"loss": float, "psnr": float, "ssim": float, "lpips": float} averaged over every
    window in `loader`.
    """
    total_loss, total_psnr, total_ssim, total_lpips, n_windows = 0.0, 0.0, 0.0, 0.0, 0
    max_noise_level = 1000
    noise_abs_max = 20
    ac = alphas_cumprod_flat.view(-1, 1, 1, 1)  # (1000, 1, 1, 1), matches generate_rollout's own reshape

    for frames, actions in loader:
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        B, T = frames.shape[:2]
        H, W = frames.shape[-2:]

        flat = rearrange(frames, "b t c h w -> (b t) c h w")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents = vae.encode(flat * 2 - 1).mean * scaling_factor
        latents = rearrange(
            latents.float(), "(b t) (h w) c -> b t c h w", t=T, h=H // vae.patch_size, w=W // vae.patch_size
        )

        # --- fixed-noise-level v-prediction loss, same formula as training ---
        t_idx = torch.full((B, T), 500, dtype=torch.long, device=device)
        noise = torch.randn_like(latents)
        ac_t = alphas_cumprod_flat[t_idx].view(B, T, 1, 1, 1)
        x_t = ac_t.sqrt() * latents + (1 - ac_t).sqrt() * noise
        v_target = ac_t.sqrt() * noise - (1 - ac_t).sqrt() * latents
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_pred = model(x_t, t_idx, actions)
        total_loss += F.mse_loss(v_pred.float(), v_target).item() * B

        # --- one-step-ahead PSNR: real context (stabilization_level), genuine DDIM chain for target ---
        context = latents[:, :-1]  # (B, T-1, C, h, w) -- all real
        target_noise = torch.clamp(torch.randn_like(latents[:, -1:]), -noise_abs_max, noise_abs_max)
        x = torch.cat([context, target_noise], dim=1)  # (B, T, C, h, w)
        noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_ctx = torch.full((B, T - 1), stabilization_level - 1, dtype=torch.long, device=device)
            t_tgt = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_tgt_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_tgt_next = torch.where(t_tgt_next < 0, t_tgt, t_tgt_next)
            t_full = torch.cat([t_ctx, t_tgt], dim=1)
            t_next_full = torch.cat([t_ctx, t_tgt_next], dim=1)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(x, t_full, actions)

            x_start = ac[t_full].sqrt() * x - (1 - ac[t_full]).sqrt() * v
            x_noise = ((1 / ac[t_full]).sqrt() * x - x_start) / (1 / ac[t_full] - 1).sqrt()
            alpha_next = ac[t_next_full].clone()
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])  # context frames never change
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

        target_latent = rearrange(x[:, -1], "b c h w -> b (h w) c")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen_target = (vae.decode(target_latent / scaling_factor) + 1) / 2  # (B, 3, H, W), [0,1]
        gen_target = torch.clamp(gen_target.float(), 0, 1)
        real_target = frames[:, -1]  # (B, 3, H, W), already [0,1]
        total_psnr += psnr_per_sample(gen_target, real_target).sum().item()
        total_ssim += ssim_metric(gen_target, real_target).item() * B
        total_lpips += lpips_metric(gen_target, real_target).item() * B

        n_windows += B

    return {
        "loss": total_loss / n_windows, "psnr": total_psnr / n_windows,
        "ssim": total_ssim / n_windows, "lpips": total_lpips / n_windows,
    }


@dataclass
class Args:
    split_path: str = "outputs/finetune_split.json"
    oasis_ckpt: str = "open-oasis/oasis500m.safetensors"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    window_len: int = 32
    batch_size: int = 4
    freeze_backbone: bool = False
    """If set, the pretrained ~608M backbone is frozen entirely (requires_grad=False, excluded
    from the optimizer) and only the new 23-dim external_cond layer trains. Added after a
    qualitative rollout comparison showed that even a very low (1e-6) full-backbone LR measurably
    eroded the pretrained model's autoregressive rollout stability (the untouched-backbone
    "before" checkpoint, despite its action layer being pure random noise, stayed visually
    coherent far longer than any full-backbone fine-tune) -- this tests whether a fully surgical,
    backbone-untouched fine-tune avoids that instability altogether."""
    unfreeze_adaln: bool = False
    """Only meaningful with freeze_backbone=True: additionally unfreezes just the *_adaLN_
    modulation layers (~203M params across the 16 blocks + final layer) -- the parts of the
    network whose job is specifically to translate the conditioning vector `c` into how
    attention/MLP behave, as opposed to the attention/MLP weights themselves which determine
    visual content/style. A middle ground between fully-frozen (stable but can't properly learn
    to use the action signal) and full-backbone (unstable, degrades visual fidelity)."""
    lr_adaln: float = 1e-5
    """LR for the unfrozen adaLN-modulation layers when unfreeze_adaln=True."""
    lr_new_layer: float = 1e-4
    """LR for the freshly-initialized 23-dim external_cond layer -- higher since it starts from
    scratch, per the plan's note that the new layer and pretrained blocks likely want different
    LRs."""
    lr_pretrained: float = 1e-6
    """LR for every other (pretrained) parameter -- much lower, to fine-tune rather than
    overwrite what the checkpoint already learned. Lowered from an initial 1e-5 after that
    setting was found to collapse rollout quality (see warmup_steps docstring for the full
    diagnosis) -- 1e-5 applied to all ~608M pretrained params with no warmup was likely too
    aggressive for a 4,248-window dataset."""
    use_oasis_native_action_layout: bool = False
    """If set, reconstructs Craftium's action directly into Oasis's OWN 25-slot ACTION_KEYS
    column layout (augment_actions_to_oasis_native_layout) instead of Craftium's raw 23-dim
    vector -- matches the exact column semantics (including real, Oasis-bucket-scaled camera
    magnitude, NOT sign -- Oasis's own pretraining camera convention is continuous, unlike
    WorldMem's) the pretrained checkpoint's action-conditioning weights were trained against. See
    craftium_action_features.py's module-level mapping table for the full derivation."""
    informed_action_init: bool = False
    """If set, initializes the reachable action-layer columns (see
    OASIS_NATIVE_INFORMED_INIT_ACTION_MAP) by copying the pretrained 25-dim checkpoint's own
    weights for those columns, instead of leaving the whole layer zero-initialized. Only
    meaningful together with use_oasis_native_action_layout=True -- the plain 23-dim bespoke
    layout has no established column-level correspondence to Oasis's own action_dim=25 layout to
    copy from."""
    warmup_steps: int = 200
    """Linear LR warmup from 0 to the target LR over this many steps, applied to BOTH parameter
    groups. Added after a first fine-tuning attempt (lr_pretrained=1e-5, no warmup, no grad
    clipping) produced a checkpoint that scored better on the held-out diffusion-loss check but
    catastrophically collapsed during actual autoregressive rollout (coherent scene -> smooth
    textureless blob by frame ~40) -- a classic signature of an early destructive update to a
    well-converged pretrained model, which AdamW's bias-corrected early steps are known to cause
    without warmup. This ALSO explains why the held-out loss check didn't catch it: that check
    only measures single-step denoising accuracy at random noise levels, not the compounding
    autoregressive-rollout behavior a large early perturbation specifically damages."""
    grad_clip_norm: float = 1.0
    """Max gradient norm (torch.nn.utils.clip_grad_norm_) -- second half of the same fix, caps
    any single step's update magnitude regardless of warmup state."""
    num_steps: int = 2000
    """Target step to train to (not "additional steps" -- e.g. resuming from step 500 with
    num_steps=3000 runs 2500 more steps, ending at 3000, not 3500)."""
    resume_from: Optional[str] = None
    """Path to a checkpoint (as saved by this script) to resume model + optimizer + step-counter
    state from, instead of starting fresh from the pretrained Oasis checkpoint. Added after
    relying on "same seed -> same trajectory" to approximate resuming was correctly called out as
    not actually guaranteed (GPU float ops aren't bit-exact reproducible even with a fixed seed) --
    this loads the real optimizer state (Adam moments, LR schedule position) rather than
    re-deriving it."""
    smoke_test_steps: Optional[int] = None
    """If set, overrides num_steps and skips checkpointing -- just times N real steps to measure
    throughput/memory, per the plan's 'timed smoke test before committing to a full run' step."""
    log_every: int = 10
    checkpoint_every: int = 500
    output_dir: str = "outputs/finetune_oasis"
    device: str = "cuda:0"
    num_workers: int = 4
    seed: int = 0

    val_every: int = 100
    """Every this many steps: compute validation loss + PSNR on the fixed "val" episode set, AND
    the identical two metrics on a fixed "train_eval" episode set (see finetune_oasis_data.py /
    scripts/make_oasis_scaling_subsets.py), logged on the same step axis so the two are directly
    comparable -- the point where they start moving in opposite directions (train_eval still
    improving, val getting worse) is the overfitting-onset signal this is for. Both use the cheap
    one-step-ahead metric (compute_eval_metrics above), not a full autoregressive rollout.
    Silently skipped (with a printed warning, once) if --split-path's json has no "val" key --
    keeps this backward-compatible with older 2-way (train/test only) split files."""
    eval_ddim_steps: int = 10
    """DDIM steps for the one-step-ahead val/train_eval PSNR check (compute_eval_metrics) --
    independent of training, which never runs a reverse-diffusion chain at all."""
    use_wandb: bool = True
    wandb_project: str = "persist-oasis-finetune"
    wandb_entity: str = "leopoldgatsing-texas-a-m-university"
    wandb_run_name: Optional[str] = None
    """Defaults to the output_dir's basename if not set."""


def main(args: Args):
    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        import wandb

    action_dim = CRAFTIUM_ACTION_DIM_OASIS_NATIVE if args.use_oasis_native_action_layout else CRAFTIUM_ACTION_DIM
    model, vae = load_oasis_for_finetuning(
        args.oasis_ckpt, args.vae_ckpt, device, action_dim=action_dim,
        informed_action_init=args.informed_action_init,
    )

    # Loaded ONCE, here, if resuming -- cached in resume_ckpt and reused later for
    # optimizer/scheduler state too (once those exist), rather than re-reading a multi-GB file
    # twice. Read this early specifically so wandb.init (right below) can reopen the ORIGINAL
    # wandb run instead of silently creating a new one -- see its comment for why that matters.
    resume_ckpt = None
    resume_wandb_run_id = None
    if args.resume_from is not None:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        resume_wandb_run_id = resume_ckpt.get("wandb_run_id")

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name or os.path.basename(os.path.normpath(args.output_dir)),
            config=vars(args),
            id=resume_wandb_run_id, resume="must" if resume_wandb_run_id else None,
        )
        if args.resume_from is not None and resume_wandb_run_id is None:
            print(f"WARNING: resuming from {args.resume_from}, but it has no saved wandb_run_id "
                  f"(saved before this feature existed?) -- this will start a NEW wandb run "
                  f"rather than continuing the original one's chart history.")

    if args.resume_from is None and args.smoke_test_steps is None:
        # Save the exact pre-training state (pretrained backbone + freshly-initialized 23-dim
        # action layer) as step_000000.pt -- the correct "before" baseline for measuring whether
        # fine-tuning helped. The original 25-dim pretrained checkpoint is NOT a valid "before"
        # baseline by itself: it can't even consume Craftium's 23-dim actions, so comparing
        # against it would conflate "the action layer was never trained" with "fine-tuning
        # didn't help" -- this step-0 checkpoint isolates the latter question cleanly. Skipped
        # when resuming -- that step-0 checkpoint already exists from the original run.
        step0_path = os.path.join(args.output_dir, "step_000000.pt")
        torch.save({
            "model": model.state_dict(), "step": 0, "args": vars(args),
            "wandb_run_id": wandb.run.id if args.use_wandb else None,
        }, step0_path)
        print(f"Saved pre-training checkpoint (pretrained backbone + fresh 23-dim action layer) to {step0_path}")

    model.train()

    dataset = CraftiumOasisDataset(
        args.split_path, "train", window_len=args.window_len,
        use_oasis_native_action_layout=args.use_oasis_native_action_layout, mode="sliding",
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )

    # Fixed val / train-eval loaders for the cheap one-step-ahead check (see Args.val_every).
    # Gracefully disabled (once-printed warning, not a crash) if --split-path doesn't have a
    # "val" key -- e.g. the older 2-way (train/test only) split files used by this session's
    # earlier single-episode overfit runs.
    with open(args.split_path) as f:
        split_keys = set(json.load(f).keys())
    run_validation = args.val_every > 0 and "val" in split_keys and "train_eval" in split_keys
    if args.val_every > 0 and not run_validation:
        print(f"WARNING: --split-path {args.split_path} has no 'val'/'train_eval' key -- "
              f"skipping validation (val_every={args.val_every} has no effect).")
    val_loader = train_eval_loader = None
    if run_validation:
        val_dataset = CraftiumOasisDataset(
            args.split_path, "val", window_len=args.window_len,
            use_oasis_native_action_layout=args.use_oasis_native_action_layout, mode="fixed",
        )
        train_eval_dataset = CraftiumOasisDataset(
            args.split_path, "train_eval", window_len=args.window_len,
            use_oasis_native_action_layout=args.use_oasis_native_action_layout, mode="fixed",
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train_eval_loader = DataLoader(
            train_eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        print(f"Validation enabled: {len(val_dataset)} val windows, {len(train_eval_dataset)} "
              f"train-eval windows, checked every {args.val_every} steps.")

    # Instantiated once and reused across every val/train-eval check (LPIPS loads a pretrained
    # AlexNet backbone -- recreating it every check would repeatedly reload those weights for no
    # reason). normalize=True on LPIPS / data_range=1.0 on SSIM since gen/real frames are [0,1],
    # not the [-1,1] LPIPS otherwise assumes.
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device) if run_validation else None
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device) \
        if run_validation else None

    new_layer_params = list(model.external_cond.parameters())
    new_layer_ids = {id(p) for p in new_layer_params}
    pretrained_params = [p for p in model.parameters() if id(p) not in new_layer_ids]

    if args.freeze_backbone:
        for p in pretrained_params:
            p.requires_grad_(False)
        param_groups = [{"params": new_layer_params, "lr": args.lr_new_layer}]
        if args.unfreeze_adaln:
            adaln_params = [
                p for n, p in model.named_parameters() if "adaLN_modulation" in n and id(p) not in new_layer_ids
            ]
            for p in adaln_params:
                p.requires_grad_(True)
            param_groups.append({"params": adaln_params, "lr": args.lr_adaln})
            print(f"unfreeze_adaln=True: also training {sum(p.numel() for p in adaln_params):,} "
                  f"adaLN-modulation params")
        optimizer = torch.optim.AdamW(param_groups)
        n_frozen = sum(p.numel() for p in pretrained_params if not p.requires_grad)
        print(f"freeze_backbone=True: {n_frozen:,} backbone params frozen, training "
              f"{sum(p.numel() for p in new_layer_params):,} action-layer params"
              + (f" + adaLN params" if args.unfreeze_adaln else ""))
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": pretrained_params, "lr": args.lr_pretrained},
                {"params": new_layer_params, "lr": args.lr_new_layer},
            ]
        )

    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / args.warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # model + wandb run were already resumed above (right after model construction); apply the
    # rest of resume_ckpt (optimizer/scheduler state, start step) now that those objects exist.
    start_step = 0
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_step = resume_ckpt["step"]
        print(f"Resumed model + optimizer + scheduler from {args.resume_from} at step {start_step}")

    betas = sigmoid_beta_schedule(1000).float().to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)  # (1000,)
    scaling_factor = 0.07843137255  # same constant generate.py uses for the VAE latent scale

    total_steps = args.smoke_test_steps if args.smoke_test_steps is not None else args.num_steps
    step = start_step
    t0 = time.time()
    step_times = []
    pbar = tqdm(total=total_steps, initial=start_step, desc="Fine-tuning Oasis")
    while step < total_steps:
        for frames, actions in loader:
            if step >= total_steps:
                break
            step_start = time.time()
            frames = frames.to(device, non_blocking=True)  # (B, T, 3, H, W)
            actions = actions.to(device, non_blocking=True)  # (B, T, 23)
            B, T = frames.shape[:2]

            with torch.no_grad():
                flat = rearrange(frames, "b t c h w -> (b t) c h w")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latents = vae.encode(flat * 2 - 1).mean * scaling_factor
                latents = rearrange(latents.float(), "(b t) (h w) c -> b t c h w", t=T, h=18, w=32)

            # Diffusion Forcing: independent random noise level per frame (see module docstring).
            t_idx = torch.randint(0, 1000, (B, T), device=device)
            noise = torch.randn_like(latents)
            ac = alphas_cumprod[t_idx].view(B, T, 1, 1, 1)
            x_t = ac.sqrt() * latents + (1 - ac).sqrt() * noise
            v_target = ac.sqrt() * noise - (1 - ac).sqrt() * latents

            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_pred = model(x_t, t_idx, actions)
                loss = F.mse_loss(v_pred.float(), v_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            step_times.append(time.time() - step_start)
            step += 1
            pbar.update(1)
            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", grad_norm=f"{grad_norm.item():.2f}")
                if args.use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/grad_norm": grad_norm.item()}, step=step)

            if run_validation and step % args.val_every == 0:
                model.eval()
                val_metrics = compute_eval_metrics(
                    model, vae, val_loader, device, alphas_cumprod, scaling_factor,
                    ssim_metric, lpips_metric, ddim_steps=args.eval_ddim_steps,
                )
                train_eval_metrics = compute_eval_metrics(
                    model, vae, train_eval_loader, device, alphas_cumprod, scaling_factor,
                    ssim_metric, lpips_metric, ddim_steps=args.eval_ddim_steps,
                )
                model.train()
                print(f"[step {step}] val: loss={val_metrics['loss']:.4f} psnr={val_metrics['psnr']:.2f}dB "
                      f"ssim={val_metrics['ssim']:.4f} lpips={val_metrics['lpips']:.4f}  "
                      f"train_eval: loss={train_eval_metrics['loss']:.4f} psnr={train_eval_metrics['psnr']:.2f}dB "
                      f"ssim={train_eval_metrics['ssim']:.4f} lpips={train_eval_metrics['lpips']:.4f}")
                if args.use_wandb:
                    wandb.log({
                        "val/loss": val_metrics["loss"], "val/psnr": val_metrics["psnr"],
                        "val/ssim": val_metrics["ssim"], "val/lpips": val_metrics["lpips"],
                        "train_eval/loss": train_eval_metrics["loss"], "train_eval/psnr": train_eval_metrics["psnr"],
                        "train_eval/ssim": train_eval_metrics["ssim"], "train_eval/lpips": train_eval_metrics["lpips"],
                    }, step=step)

            if args.smoke_test_steps is None and step % args.checkpoint_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"step_{step:06d}.pt")
                torch.save({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "args": vars(args),
                    "wandb_run_id": wandb.run.id if args.use_wandb else None,
                }, ckpt_path)
                if args.use_wandb:
                    wandb.log({"checkpoint_step": step}, step=step)

    pbar.close()
    elapsed = time.time() - t0
    mean_step_time = sum(step_times) / len(step_times)
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    print(f"\n=== Run summary (for the paper's conditions log) ===")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Steps: {step}, batch_size={args.batch_size}, window_len={args.window_len}")
    print(f"Total wall-clock: {elapsed:.1f}s, mean step time: {mean_step_time:.3f}s/step "
          f"({1/mean_step_time:.2f} steps/s)")
    print(f"Peak GPU memory allocated: {peak_mem_gb:.2f} GB")
    print(f"Final loss: {loss.item():.4f}")

    if args.smoke_test_steps is None:
        final_path = os.path.join(args.output_dir, "final.pt")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "step": step, "args": vars(args),
            "wandb_run_id": wandb.run.id if args.use_wandb else None,
        }, final_path)
        print(f"Saved final checkpoint to {final_path}")

    if args.use_wandb:
        wandb.log({
            "summary/peak_gpu_mem_gb": peak_mem_gb, "summary/mean_step_time_s": mean_step_time,
        }, step=step)
        wandb.finish()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
