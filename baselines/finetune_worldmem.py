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
from craftium_action_features import (  # noqa: E402
    CRAFTIUM_ACTION_DIM_ENRICHED,
    CRAFTIUM_ACTION_DIM_WORLDMEM_NATIVE,
    WORLDMEM_NATIVE_INFORMED_INIT_ACTION_MAP,
)

CRAFTIUM_ACTION_DIM = 23


class GatedActionEmbed(torch.nn.Module):
    """Same idea as finetune_oasis.py's GatedActionEmbed: drop-in replacement for the plain
    `nn.Linear(action_cond_dim, hidden_size)` external_cond layer, with a learnable scalar gate
    controlling the action signal's overall magnitude before it's added into the shared
    conditioning vector `c`.

    BUG FIX (see finetune_oasis.py's GatedActionEmbed docstring for the full derivation and the
    empirical confirmation -- loading trained checkpoints directly and finding `external_cond`'s
    output byte-identical for real/random/zero actions after thousands of steps, in every
    fine-tuning run of both models): `gate` previously started at exactly 0, matching `linear`'s
    own zero-initialized weight/bias. `output = gate * (W @ action + b)` is a product of two
    independently zero-initialized learnable quantities, so by the product rule every gradient
    (`d(gate)`, `d(W)`, `d(b)`) is exactly 0 at that point -- a permanent fixed point, not
    something more training fixes. Fix: initialize `gate` to a nonzero value while keeping
    `linear` zero-initialized -- output at step 0 is still exactly 0 (`linear`'s output is 0, so
    `0 * gate == 0` regardless of `gate`), preserving "no initial perturbation," but now
    `d(loss)/d(linear.weight) = gate * action != 0`, so `linear` can start learning immediately,
    and `gate`'s own gradient becomes nonzero as soon as it does."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(action_dim, hidden_size)
        self.gate = torch.nn.Parameter(torch.ones(1))

    def forward(self, action):
        return self.linear(action) * self.gate


# Craftium idx -> Minecraft/VPT idx, restricted to the subset that is BOTH a clean 1:1 semantic
# match AND actually exercised by WorldMem's own training data (its data generator only ever
# issues forward/back/left/right/camera/drop/hotbar.1 -- see docs/action_space_comparison.md
# section 1 and section 4). jump/sneak/aux1/dig/place/slot_2-9 are excluded even where a clean
# semantic match exists (e.g. craftium `jump`->minecraft `jump`), because WorldMem's own pretrained
# action layer never received a real training signal for those columns -- their pretrained weights
# are uninformative, copying them would inject noise rather than knowledge. Camera is excluded for
# a different reason: Minecraft's camera is one continuous scalar per axis, Craftium's is two
# discrete fixed-step booleans -- a structurally different signal, not a matter of reindexing.
INFORMED_INIT_ACTION_MAP = {
    0: 11,   # forward -> forward
    1: 12,   # backward -> back
    2: 13,   # left -> left
    3: 14,   # right -> right
    9: 24,   # drop -> drop
    10: 2,   # slot_1 -> hotbar.1
}


def load_worldmem_for_finetuning(
    config_path: str, device: str, zero_init_action_layer: bool = True, gated_action: bool = True,
    informed_action_init: bool = False, action_dim: int = CRAFTIUM_ACTION_DIM,
    informed_init_map: dict = INFORMED_INIT_ACTION_MAP,
):
    """Same instantiation path as eval_worldmem_zeroshot.py:load_worldmem, but with
    action_cond_dim overridden to 23 before construction, pose-prediction disabled (out of scope
    for this pass -- see module docstring), and the mismatched action-layer checkpoint keys
    dropped/reinitialized instead of loaded, matching finetune_oasis.py's approach.

    informed_action_init: instead of leaving the new action layer entirely zero-initialized, copy
    the pretrained 25-dim checkpoint's own weight COLUMNS for the dims in informed_init_map (source
    idx -> pretrained/Minecraft idx) into the corresponding new columns (nn.Linear's weight is
    (hidden, in_features), so one column = one input dimension's learned projection), and copy the
    bias directly (shape (hidden,), independent of input dim). Every other column stays
    zero-initialized exactly as before.

    informed_init_map: defaults to INFORMED_INIT_ACTION_MAP (the 23-dim bespoke-append layout's
    6-entry map). Pass WORLDMEM_NATIVE_INFORMED_INIT_ACTION_MAP instead when action_dim ==
    CRAFTIUM_ACTION_DIM_WORLDMEM_NATIVE (that map is an identity map -- source and destination
    columns already coincide, since augment_actions_to_worldmem_native_layout places every value
    directly in WorldMem's own ACTION_KEYS column position).

    Trade-off worth knowing: this breaks GatedActionEmbed's "output is exactly 0 at step 0"
    property for whichever mapped actions are active on frame 0 (movement keys are pressed on most
    frames), since the gate no longer has an all-zero linear() to multiply against. That's an
    intentional exchange of "no perturbation" safety for "start with genuinely relevant pretrained
    signal on the columns most likely to already mean something to the backbone" -- untested,
    which is why this is an opt-in flag rather than the new default."""
    from algorithms.worldmem import WorldMemMinecraft

    cfg = OmegaConf.load(config_path)
    cfg.action_cond_dim = action_dim
    cfg.require_pose_prediction = False
    worldmem = WorldMemMinecraft(cfg)

    action_layer = worldmem.diffusion_model.model.external_cond
    if zero_init_action_layer:
        torch.nn.init.zeros_(action_layer.weight)
        torch.nn.init.zeros_(action_layer.bias)

    diffusion_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.diffusion_path.split("/")[:2]),
        filename="/".join(cfg.diffusion_path.split("/")[2:]),
    )
    ckpt = torch.load(diffusion_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    if informed_action_init:
        pretrained_weight = state_dict.get("model.external_cond.weight")
        pretrained_bias = state_dict.get("model.external_cond.bias")
        if pretrained_weight is None or pretrained_weight.shape[1] != 25:
            raise ValueError(
                f"informed_action_init=True but couldn't find a 25-dim pretrained external_cond "
                f"weight in the checkpoint (got {None if pretrained_weight is None else pretrained_weight.shape})."
            )
        with torch.no_grad():
            for src_idx, minecraft_idx in informed_init_map.items():
                action_layer.weight[:, src_idx].copy_(pretrained_weight[:, minecraft_idx])
            action_layer.bias.copy_(pretrained_bias)
        print(f"informed_action_init=True: copied {len(informed_init_map)} pretrained action "
              f"columns {informed_init_map} into the new {action_dim}-dim layer; bias copied in full.")

    if gated_action:
        gated = GatedActionEmbed(action_dim, action_layer.out_features)
        gated.linear.weight.data.copy_(action_layer.weight.data)
        gated.linear.bias.data.copy_(action_layer.bias.data)
        worldmem.diffusion_model.model.external_cond = gated

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
    informed_action_init: bool = False
    """If set, initializes the 6 action-layer columns with a clean 1:1 Craftium<->Minecraft
    semantic match AND real training exposure (forward/back/left/right/drop/hotbar.1) by copying
    the pretrained 25-dim checkpoint's own weights for those columns, instead of leaving the whole
    layer zero-initialized. See load_worldmem_for_finetuning's docstring for the full mapping and
    the trade-off (breaks the zero-output-at-step-0 property for these 6 actions)."""
    use_clean_camera_features: bool = False
    """If set, appends 2 extra action columns (yaw_delta, pitch_delta) derived from
    player_yaw/player_pitch, gated on each axis's own action being active -- verified to be a
    clean, near-exact fixed-magnitude signal (measured 6.40 deg/step) unlike Craftium's raw
    boolean camera flags, which map to a variable real turn amount (see
    craftium_action_features.py and docs/action_space_empirical_report.md). Action dim becomes 25
    instead of 23 when enabled. Mutually exclusive with use_worldmem_native_action_layout."""
    use_worldmem_native_action_layout: bool = False
    """If set, reconstructs Craftium's action directly into WorldMem's OWN 25-slot ACTION_KEYS
    column layout (augment_actions_to_worldmem_native_layout) instead of appending 2 bespoke
    columns after Craftium's raw 23 -- matches the exact column semantics the pretrained
    checkpoint's action-conditioning weights were trained against (see
    craftium_action_features.py's module-level mapping table and
    WORLDMEM_NATIVE_INFORMED_INIT_ACTION_MAP). Mutually exclusive with use_clean_camera_features.
    Per the Hypothesis A finding (single-seed causality verdicts are unreliable measurement noise,
    not a real training-dynamics signal -- see eval_worldmem_pose_causality_cleancam.py), any
    future evaluation of this layout must use --num-seeds >= 5, not a single seed."""
    lr_adaln: float = 1e-5
    lr_new_layer: float = 1e-4
    lr_pretrained: float = 1e-6
    lr_pose: float = 1e-5
    """LR for the pose-conditioning params (pose_cond_mlp, temporal_pose_cond_mlp,
    r_adaLN_modulation) -- unlike the new 23-dim action layer, these are INHERITED from
    pretraining, not new, so they were previously bundled into lr_pretrained (1e-6) along with the
    rest of the backbone. Found (via a real/shuffled/zero pose ablation, on both level_053 and
    level_006) that feeding the model its OWN real, correct pose consistently scores WORSE than
    feeding it wrong or zero pose -- ruled out as an eval-methodology gap (tested WorldMem's own
    documented warm-start-memory protocol directly, no improvement), a geometric bug (tested
    Craftium's actual anisotropic FOV against the single-scalar `focal_length` config value the
    Plucker conversion uses, corrected it, no improvement), and fine-tuning-induced corruption
    (compared pose_cond_mlp/r_adaLN_modulation's weights before vs. after fine-tuning: <2% relative
    change -- and a broader check found the SAME near-zero relative change for the spatial/temporal
    blocks too, meaning at lr_pretrained=1e-6 essentially nothing in the backbone moves much in a
    few thousand steps). The remaining, most direct explanation: pose conditioning genuinely needs
    Craftium-specific recalibration (real Minecraft pose data differs from Craftium's in scale/FOV
    convention) but was never given the chance, since it shares the action layer's need to adapt
    without sharing its boosted learning rate. This flag tests that directly."""
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
    assert not (args.use_clean_camera_features and args.use_worldmem_native_action_layout), (
        "use_clean_camera_features and use_worldmem_native_action_layout are mutually exclusive "
        "action-layout choices -- pick one."
    )
    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_clean_camera_features:
        action_dim = CRAFTIUM_ACTION_DIM_ENRICHED
        informed_init_map = INFORMED_INIT_ACTION_MAP
    elif args.use_worldmem_native_action_layout:
        action_dim = CRAFTIUM_ACTION_DIM_WORLDMEM_NATIVE
        informed_init_map = WORLDMEM_NATIVE_INFORMED_INIT_ACTION_MAP
    else:
        action_dim = CRAFTIUM_ACTION_DIM
        informed_init_map = INFORMED_INIT_ACTION_MAP
    worldmem = load_worldmem_for_finetuning(
        args.config_path, device, informed_action_init=args.informed_action_init, action_dim=action_dim,
        informed_init_map=informed_init_map,
    )

    if args.resume_from is None and args.smoke_test_steps is None:
        step0_path = os.path.join(args.output_dir, "step_000000.pt")
        torch.save({"model": worldmem.diffusion_model.state_dict(), "step": 0, "args": vars(args)}, step0_path)
        print(f"Saved pre-training checkpoint to {step0_path}")

    worldmem.diffusion_model.train()

    dataset = CraftiumWorldMemDataset(
        args.split_path, "train", n_frames=args.n_frames, memory_condition_length=args.memory_condition_length,
        seed=args.seed, use_clean_camera_features=args.use_clean_camera_features,
        use_worldmem_native_action_layout=args.use_worldmem_native_action_layout,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )

    new_layer_params = list(worldmem.diffusion_model.model.external_cond.parameters())
    new_layer_ids = {id(p) for p in new_layer_params}
    pretrained_params = [p for p in worldmem.diffusion_model.parameters() if id(p) not in new_layer_ids]

    # Pose-conditioning params (see Args.lr_pose docstring): inherited from pretraining, not new,
    # but found -- via a real/shuffled/zero pose ablation -- to need Craftium-specific
    # recalibration just like the action layer does, so they're split out of pretrained_params
    # into their own faster-learning group instead of moving at the same glacial backbone pace.
    pose_param_names = ("pose_cond_mlp", "temporal_pose_cond_mlp", "r_adaLN_modulation")
    pose_params = [
        p for n, p in worldmem.diffusion_model.named_parameters()
        if any(pn in n for pn in pose_param_names) and id(p) not in new_layer_ids
    ]
    pose_param_ids = {id(p) for p in pose_params}
    # backbone_params = pretrained_params minus pose_params, so the three optimizer groups below
    # (backbone/pose/new-layer) partition every trainable parameter exactly once, no overlap.
    backbone_params = [p for p in pretrained_params if id(p) not in pose_param_ids]

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
            print(f"unfreeze_adaln=True: also training {sum(p.numel() for p in adaln_params):,} adaLN-modulation params")
        optimizer = torch.optim.AdamW(param_groups)
        n_frozen = sum(p.numel() for p in pretrained_params if not p.requires_grad)
        print(f"freeze_backbone=True: {n_frozen:,} backbone params frozen, training "
              f"{sum(p.numel() for p in new_layer_params):,} action-layer params"
              + (" + adaLN params" if args.unfreeze_adaln else ""))
    else:
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr_pretrained},
            {"params": pose_params, "lr": args.lr_pose},
            {"params": new_layer_params, "lr": args.lr_new_layer},
        ])
        print(f"freeze_backbone=False: {sum(p.numel() for p in backbone_params):,} backbone params @ lr={args.lr_pretrained}, "
              f"{sum(p.numel() for p in pose_params):,} pose-conditioning params @ lr={args.lr_pose}, "
              f"{sum(p.numel() for p in new_layer_params):,} action-layer params @ lr={args.lr_new_layer}")

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

    print("\n=== Run summary (for the paper's conditions log) ===")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Steps: {step}, batch_size={args.batch_size}, n_frames={args.n_frames}, "
          f"memory_condition_length={args.memory_condition_length}")
    print(f"Total wall-clock: {elapsed:.1f}s, mean step time: {mean_step_time:.3f}s/step ("
          f"{1/mean_step_time:.2f} steps/s)")
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
