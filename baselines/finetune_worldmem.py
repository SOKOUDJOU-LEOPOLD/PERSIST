"""Fine-tune WorldMem's diffusion model on Craftium data, swapping its 25-dim VPT-style
action-conditioning layer for a 23-dim layer matching Craftium's native action space -- same
model-surgery pattern as finetune_oasis.py, applied to WorldMem's (structurally identical)
`nn.Linear(action_cond_dim, hidden_size)` action-conditioning layer.

Reuses WorldMem's own `WorldMemMinecraft` class and its real `training_step()` (encode -> sample
per-frame noise levels -> diffusion loss) directly -- instantiated the same way
`baselines/eval_worldmem_zeroshot.py:load_worldmem` already does for inference (plain
`OmegaConf.load` + direct construction, no Hydra CLI needed) -- rather than routing through
`main.py`'s Hydra+Lightning entry point, which would need new Hydra config files and doesn't
buy anything for this single-process fine-tuning run.

Uses CraftiumWorldMemDataset with the real FOV-overlap+recency greedy memory selection (the exact
function `validation_step`/`interactive()` use at generation time, `_generate_condition_indices`)
and proper pose normalization (see that file's docstring) -- restored after two earlier fidelity
gaps from the original design: pure random selection, then a different ("smart" but not the one
generation actually uses) pos_range/angle_range distance-threshold selection.

`--freeze-backbone`/`--unfreeze-adaln`: mirrors the fix that resolved Oasis's fine-tuning
instability -- training the ENTIRE ~945M diffusion_model (this script's original behavior)
produced severe RGB-striped corruption during autoregressive rollout, the same class of failure
full-backbone fine-tuning caused for Oasis. Freezing the bulk of the backbone and training only
the action layer (+ optionally the s_/t_/r_/final_layer.adaLN_modulation layers -- the parts of
the network whose job is to translate the conditioning vector into attention/MLP behavior, not
the attention/MLP weights themselves) is the same targeted-adaptation strategy that fixed Oasis.

Example:
    cd baselines/WorldMem
    python ../finetune_worldmem.py --freeze-backbone --unfreeze-adaln --smoke-test-steps 20
"""
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorldMem"))

from finetune_worldmem_data import CraftiumWorldMemDataset  # noqa: E402

CRAFTIUM_ACTION_DIM = 23


class GatedActionEmbed(torch.nn.Module):
    """Same idea as finetune_oasis.py's GatedActionEmbed: drop-in replacement for the plain
    `nn.Linear(action_cond_dim, hidden_size)` external_cond layer, with a learnable,
    zero-initialized scalar gate controlling the action signal's overall magnitude before it's
    added into the shared conditioning vector `c`."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(action_dim, hidden_size)
        self.gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, action):
        return self.linear(action) * self.gate


def load_worldmem_for_finetuning(
    config_path: str, device: str, zero_init_action_layer: bool = True, gated_action: bool = True,
):
    """Same instantiation path as eval_worldmem_zeroshot.py:load_worldmem, but with
    action_cond_dim overridden to 23 before construction, pose-prediction disabled (out of scope
    for this pass -- see module docstring), and the mismatched action-layer checkpoint keys
    dropped/reinitialized instead of loaded, matching finetune_oasis.py's approach."""
    from algorithms.worldmem import WorldMemMinecraft

    cfg = OmegaConf.load(config_path)
    cfg.action_cond_dim = CRAFTIUM_ACTION_DIM
    cfg.require_pose_prediction = False
    worldmem = WorldMemMinecraft(cfg)

    action_layer = worldmem.diffusion_model.model.external_cond
    if zero_init_action_layer:
        torch.nn.init.zeros_(action_layer.weight)
        torch.nn.init.zeros_(action_layer.bias)
    if gated_action:
        gated = GatedActionEmbed(CRAFTIUM_ACTION_DIM, action_layer.out_features)
        gated.linear.weight.data.copy_(action_layer.weight.data)
        gated.linear.bias.data.copy_(action_layer.bias.data)
        worldmem.diffusion_model.model.external_cond = gated

    diffusion_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.diffusion_path.split("/")[:2]),
        filename="/".join(cfg.diffusion_path.split("/")[2:]),
    )
    ckpt = torch.load(diffusion_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    mismatched = {k for k in state_dict if k.startswith("model.external_cond.")}
    filtered = {k: v for k, v in state_dict.items() if k not in mismatched}
    missing, unexpected = worldmem.diffusion_model.load_state_dict(filtered, strict=False)
    # See finetune_oasis.py's identical comment: with gated_action=True, the model's own
    # external_cond parameter names are external_cond.linear.{weight,bias}/external_cond.gate,
    # not external_cond.{weight,bias} -- matched by prefix rather than exact name for that reason.
    expected_missing = {k for k in missing if "external_cond" in k} | {
        k for k in missing if k.endswith("rotary_emb.freqs")
    }
    unexpected_missing = set(missing) - expected_missing
    assert not unexpected_missing, f"Unexpected missing keys (checkpoint/architecture mismatch): {unexpected_missing}"
    assert not unexpected, f"Unexpected extra keys in checkpoint: {unexpected}"

    vae_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.vae_path.split("/")[:2]), filename="/".join(cfg.vae_path.split("/")[2:])
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    worldmem.vae.load_state_dict(vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt, strict=True)
    worldmem.vae.eval()
    for p in worldmem.vae.parameters():
        p.requires_grad_(False)

    return worldmem.to(device)


@dataclass
class Args:
    split_path: str = "../outputs/finetune_split.json"
    config_path: str = "configurations/huggingface.yaml"
    n_frames: int = 8
    memory_condition_length: int = 8
    batch_size: int = 2
    freeze_backbone: bool = False
    """If set, the pretrained ~945M diffusion_model backbone is frozen entirely and only the new
    23-dim action layer trains -- see module docstring for why this was added (mirrors the fix
    that resolved Oasis's full-backbone-fine-tuning instability)."""
    unfreeze_adaln: bool = False
    """Only meaningful with freeze_backbone=True: additionally unfreezes the s_/t_/r_/
    final_layer.adaLN_modulation layers (the parts of the network that translate the conditioning
    vector into attention/MLP behavior) -- same targeted middle ground used for Oasis."""
    lr_adaln: float = 1e-5
    lr_new_layer: float = 1e-4
    lr_pretrained: float = 1e-6
    warmup_steps: int = 200
    grad_clip_norm: float = 1.0
    resume_from: Optional[str] = None
    num_steps: int = 3000
    smoke_test_steps: Optional[int] = None
    log_every: int = 10
    checkpoint_every: int = 500
    output_dir: str = "../outputs/finetune_worldmem"
    device: str = "cuda:0"
    num_workers: int = 4
    seed: int = 0


def main(args: Args):
    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    worldmem = load_worldmem_for_finetuning(args.config_path, device)

    if args.resume_from is None and args.smoke_test_steps is None:
        step0_path = os.path.join(args.output_dir, "step_000000.pt")
        torch.save({"model": worldmem.diffusion_model.state_dict(), "step": 0, "args": vars(args)}, step0_path)
        print(f"Saved pre-training checkpoint to {step0_path}")

    worldmem.diffusion_model.train()

    dataset = CraftiumWorldMemDataset(
        args.split_path, "train", n_frames=args.n_frames, memory_condition_length=args.memory_condition_length,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )

    new_layer_params = list(worldmem.diffusion_model.model.external_cond.parameters())
    new_layer_ids = {id(p) for p in new_layer_params}
    pretrained_params = [p for p in worldmem.diffusion_model.parameters() if id(p) not in new_layer_ids]

    if args.freeze_backbone:
        for p in pretrained_params:
            p.requires_grad_(False)
        param_groups = [{"params": new_layer_params, "lr": args.lr_new_layer}]
        if args.unfreeze_adaln:
            adaln_params = [
                p for n, p in worldmem.diffusion_model.named_parameters()
                if "adaLN_modulation" in n and id(p) not in new_layer_ids
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
              + (" + adaLN params" if args.unfreeze_adaln else ""))
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
        worldmem.diffusion_model.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_step = resume_ckpt["step"]
        print(f"Resumed model + optimizer + scheduler from {args.resume_from} at step {start_step}")

    total_steps = args.smoke_test_steps if args.smoke_test_steps is not None else args.num_steps
    step = start_step
    t0 = time.time()
    step_times = []
    pbar = tqdm(total=total_steps, initial=start_step, desc="Fine-tuning WorldMem")
    while step < total_steps:
        for video, actions, poses, timestamp in loader:
            if step >= total_steps:
                break
            step_start = time.time()
            batch = (
                video.to(device, non_blocking=True),
                actions.to(device, non_blocking=True),
                poses.to(device, non_blocking=True),
                timestamp.to(device, non_blocking=True),
            )

            loss_dict = worldmem.training_step(batch, step)
            loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(worldmem.diffusion_model.parameters(), args.grad_clip_norm)
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
                    "model": worldmem.diffusion_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "args": vars(args),
                }, ckpt_path)

    pbar.close()
    elapsed = time.time() - t0
    mean_step_time = sum(step_times) / len(step_times)
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    print(f"\n=== Run summary (for the paper's conditions log) ===")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Steps: {step}, batch_size={args.batch_size}, n_frames={args.n_frames}, "
          f"memory_condition_length={args.memory_condition_length}")
    print(f"Total wall-clock: {elapsed:.1f}s, mean step time: {mean_step_time:.3f}s/step "
          f"({1/mean_step_time:.2f} steps/s)")
    print(f"Peak GPU memory allocated: {peak_mem_gb:.2f} GB")
    print(f"Final loss: {loss.item():.4f}")

    if args.smoke_test_steps is None:
        final_path = os.path.join(args.output_dir, "final.pt")
        torch.save({
            "model": worldmem.diffusion_model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "step": step, "args": vars(args),
        }, final_path)
        print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
