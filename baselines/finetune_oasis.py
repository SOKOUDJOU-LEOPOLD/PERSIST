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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from finetune_oasis_data import CraftiumOasisDataset  # noqa: E402

CRAFTIUM_ACTION_DIM = 23


class GatedActionEmbed(torch.nn.Module):
    """Drop-in replacement for DiT's plain `nn.Linear(action_dim, hidden_size)` external_cond
    layer: same call signature (`self(action) -> (..., hidden_size)`), but the output is scaled
    by a learnable, zero-initialized scalar gate before being returned. Since dit.py's forward
    does `c += self.external_cond(external_cond)` unconditionally, this needs no changes to
    dit.py at all -- it's substituted in after construction (see load_oasis_for_finetuning).

    Motivation: even with a zero-initialized linear layer, once trained its raw output is added
    into `c` completely unscaled -- nothing stops training from growing that contribution to a
    magnitude that measurably shifts the shared conditioning signal for EVERY frame, degrading
    base visual fidelity even while the action-following objective improves (this is the
    hypothesized cause of the color/lighting drift seen in every fine-tuning attempt so far). A
    learnable gate lets the model control the action signal's magnitude explicitly and gradually,
    the same "zero-init gate" idea already used elsewhere in this project family (see
    baselines/WorldMem/experiments/exp_base.py's zero_init_gate option) but applied as an
    explicit multiplicative control rather than just a zero starting point."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(action_dim, hidden_size)
        self.gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, action):
        return self.linear(action) * self.gate


def build_dit_architecture(gated_action: bool = True, zero_init_action_layer: bool = True):
    """Construct the 23-dim-action DiT architecture only (no checkpoint weights loaded) --
    shared by load_oasis_for_finetuning (training) and every generation/eval script that needs to
    load one of our own fine-tuned checkpoints, so the architecture (in particular, whether
    external_cond is a plain nn.Linear or a GatedActionEmbed) always matches whatever the
    checkpoint being loaded was actually saved with."""
    from dit import DiT

    model = DiT(patch_size=2, hidden_size=1024, depth=16, num_heads=16, external_cond_dim=CRAFTIUM_ACTION_DIM)
    if zero_init_action_layer:
        torch.nn.init.zeros_(model.external_cond.weight)
        torch.nn.init.zeros_(model.external_cond.bias)
    if gated_action:
        gated = GatedActionEmbed(CRAFTIUM_ACTION_DIM, model.external_cond.out_features)
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
    model = build_dit_architecture(gated_action=gated_action)
    model.load_state_dict(state_dict)
    return model.to(device).eval(), ckpt.get("step")


def load_oasis_for_finetuning(
    oasis_ckpt: str, vae_ckpt: str, device: str, zero_init_action_layer: bool = True, gated_action: bool = True,
):
    """Load Oasis's DiT with a freshly-initialized 23-dim action layer, everything else from the
    pretrained checkpoint. See module docstring / the plan's "model surgery" step.

    `zero_init_action_layer`: zero-initializes the new layer's weight AND bias (rather than
    nn.Linear's default random init), so at step 0 `external_cond(actions) == 0` for every
    action -- i.e. training starts from the literal pretrained model's unmodified behavior
    (c += 0) and has to learn to incorporate the action signal, rather than immediately injecting
    a random/systematically-biased offset into the shared conditioning vector `c`.

    `gated_action`: wraps the action layer in GatedActionEmbed (see above) instead of a plain
    nn.Linear -- adds an explicit, learnable, zero-initialized scalar controlling the action
    signal's overall magnitude, on top of zero_init_action_layer's per-weight zero-init."""
    from vae import VAE_models

    model = build_dit_architecture(gated_action=gated_action, zero_init_action_layer=zero_init_action_layer)
    pretrained = load_file(oasis_ckpt)
    mismatched = {k for k in pretrained if k.startswith("external_cond.")}
    filtered = {k: v for k, v in pretrained.items() if k not in mismatched}
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


def main(args: Args):
    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    model, vae = load_oasis_for_finetuning(args.oasis_ckpt, args.vae_ckpt, device)

    if args.resume_from is None and args.smoke_test_steps is None:
        # Save the exact pre-training state (pretrained backbone + freshly-initialized 23-dim
        # action layer) as step_000000.pt -- the correct "before" baseline for measuring whether
        # fine-tuning helped. The original 25-dim pretrained checkpoint is NOT a valid "before"
        # baseline by itself: it can't even consume Craftium's 23-dim actions, so comparing
        # against it would conflate "the action layer was never trained" with "fine-tuning
        # didn't help" -- this step-0 checkpoint isolates the latter question cleanly. Skipped
        # when resuming -- that step-0 checkpoint already exists from the original run.
        step0_path = os.path.join(args.output_dir, "step_000000.pt")
        torch.save({"model": model.state_dict(), "step": 0, "args": vars(args)}, step0_path)
        print(f"Saved pre-training checkpoint (pretrained backbone + fresh 23-dim action layer) to {step0_path}")

    model.train()

    dataset = CraftiumOasisDataset(args.split_path, "train", window_len=args.window_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )

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

    start_step = 0
    if args.resume_from is not None:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
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

            if args.smoke_test_steps is None and step % args.checkpoint_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"step_{step:06d}.pt")
                torch.save({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "args": vars(args),
                }, ckpt_path)

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
        }, final_path)
        print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
