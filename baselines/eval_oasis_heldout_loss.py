"""Cheapest possible check for whether Oasis fine-tuning did anything: compare the
diffusion-forcing loss (the exact training objective, see finetune_oasis.py's docstring) of two
checkpoints on the SAME fixed batch of held-out (test-split) windows, corrupted with the SAME
fixed noise/timestep seed. No sampling/generation, no I3D, no video writing -- one forward pass
per checkpoint.

"Before" is step_000000.pt (pretrained backbone + freshly-initialized 23-dim action layer, saved
by finetune_oasis.py right before training starts), NOT the original 25-dim checkpoint -- the
original checkpoint can't even consume Craftium's 23-dim actions, so it isn't a valid baseline
for isolating "did fine-tuning help" (see finetune_oasis.py's comment at the step_000000.pt save
site for the full reasoning).

Example:
    cd baselines
    python eval_oasis_heldout_loss.py \
        --before-ckpt ../outputs/finetune_oasis/step_000000.pt \
        --after-ckpt ../outputs/finetune_oasis/final.pt
"""
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from finetune_oasis import CRAFTIUM_ACTION_DIM, load_finetuned_dit, sigmoid_beta_schedule  # noqa: E402
from finetune_oasis_data import CraftiumOasisDataset  # noqa: E402


@dataclass
class Args:
    split_path: str = "../outputs/finetune_split.json"
    before_ckpt: str = "../outputs/finetune_oasis/step_000000.pt"
    after_ckpt: str = "../outputs/finetune_oasis/final.pt"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    window_len: int = 32
    num_windows: int = 16
    """How many fixed held-out test-split windows to evaluate on."""
    eval_seed: int = 1234
    """Seed for both which windows are picked and the noise/timestep corruption -- fixed so both
    checkpoints see byte-identical inputs."""
    device: str = "cuda:0"


load_model_at = load_finetuned_dit  # auto-detects gated vs. plain external_cond


def main(args: Args):
    device = args.device
    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    dataset = CraftiumOasisDataset(args.split_path, "test", window_len=args.window_len)
    print(f"Held-out test set: {len(dataset.level_ids)} episodes, {len(dataset)} total windows")

    g = torch.Generator().manual_seed(args.eval_seed)
    idxs = torch.randperm(len(dataset), generator=g)[: args.num_windows].tolist()
    frames_list, actions_list = zip(*[dataset[i] for i in idxs])
    frames = torch.stack(frames_list).to(device)  # (N, T, 3, H, W)
    actions = torch.stack(actions_list).to(device)  # (N, T, 23)
    N, T = frames.shape[:2]

    betas = sigmoid_beta_schedule(1000).float().to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    scaling_factor = 0.07843137255

    with torch.no_grad():
        flat = rearrange(frames, "n t c h w -> (n t) c h w")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents = vae.encode(flat * 2 - 1).mean * scaling_factor
        latents = rearrange(latents.float(), "(n t) (h w) c -> n t c h w", t=T, h=18, w=32)

    # Fixed corruption, shared across both checkpoints (separate generator so picking windows
    # above doesn't perturb this draw).
    g2 = torch.Generator(device=device).manual_seed(args.eval_seed + 1)
    t_idx = torch.randint(0, 1000, (N, T), generator=g2, device=device)
    noise = torch.randn(latents.shape, generator=g2, device=device)
    ac = alphas_cumprod[t_idx].view(N, T, 1, 1, 1)
    x_t = ac.sqrt() * latents + (1 - ac).sqrt() * noise
    v_target = ac.sqrt() * noise - (1 - ac).sqrt() * latents

    results = {}
    for label, ckpt_path in [("before", args.before_ckpt), ("after", args.after_ckpt)]:
        model, step = load_model_at(ckpt_path, device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_pred = model(x_t, t_idx, actions)
            loss = F.mse_loss(v_pred.float(), v_target).item()
        results[label] = loss
        print(f"{label:>6} (ckpt step={step}, {ckpt_path}): held-out diffusion loss = {loss:.5f}")
        del model
        torch.cuda.empty_cache()

    delta = results["before"] - results["after"]
    pct = 100 * delta / results["before"]
    print(f"\n=== Held-out loss comparison (for the paper) ===")
    print(f"num_windows={N} (from {len(dataset.level_ids)} test episodes), window_len={T}, "
          f"eval_seed={args.eval_seed}")
    print(f"before: {results['before']:.5f}  after: {results['after']:.5f}  "
          f"delta: {delta:+.5f} ({pct:+.1f}%)")
    if delta > 0:
        print("Fine-tuning REDUCED held-out loss -- evidence it learned something.")
    else:
        print("Fine-tuning did NOT reduce held-out loss -- investigate before proceeding to "
              "qualitative/FVD checks.")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
