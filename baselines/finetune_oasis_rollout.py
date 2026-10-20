"""Shared DDIM rollout function for a fine-tuned (23-dim action) Oasis checkpoint, driven by a
real Craftium seed frame + real action sequence. Factored out of finetune_oasis_generate.py so
both the standalone qualitative-video script and the FVD evaluation script
(eval_oasis_finetuned_fvd.py) use the exact same generation code -- no risk of the two diverging.

DDIM loop copied from baselines/open-oasis/generate.py (see that file / finetune_oasis_generate.py
for the original, single-episode, video-writing version this was factored out of).

Generation is autoregressive (frame t+1 depends on frame t), so there is no way to parallelize
across TIME within one episode -- the only parallelism axis is across independent episodes.
generate_rollout_batch exploits that: the DDIM loop below was already shaped generically around a
leading batch dim (B = x.shape[0] used throughout every tensor op), so batching several episodes'
seed frames + actions together and running them through the SAME DDIM step in lockstep needs no
change to the sampling math itself -- it only needed a batched entry/exit point, since the
single-episode generate_rollout hardcoded B=1 via seed_frame[None,None] and a final x[0] index.
generate_rollout is now a thin wrapper around generate_rollout_batch (unsqueeze in, index 0 out),
so every existing single-episode caller is unaffected.
"""
import torch
from einops import rearrange
from torch import autocast
from tqdm import tqdm

from finetune_oasis import sigmoid_beta_schedule


@torch.no_grad()
def generate_rollout_batch(
    model,
    vae,
    seed_frames: torch.Tensor,
    actions: torch.Tensor,
    num_frames: int,
    device: str,
    n_prompt_frames: int = 1,
    ddim_steps: int = 10,
    desc: str = "Generating",
    show_progress: bool = True,
    decode_chunk_size: int = 64,
) -> torch.Tensor:
    """Batched version of generate_rollout: generates B independent episodes in lockstep (same
    DDIM noise schedule -- it doesn't depend on content), one forward pass per DDIM step for the
    whole batch instead of one pass per episode. Use this instead of calling generate_rollout in a
    per-episode loop when generating many episodes on one GPU -- it amortizes the DiT model's
    compute across more samples per step, which matters because a single episode alone likely
    doesn't saturate an A100's compute.

    Args:
        seed_frames: (B, 3, H, W) float tensor in [0, 1].
        actions: (B, T, 23) float tensor, Craftium's native action vector, ALREADY shifted by one
            frame per episode (see generate_rollout's docstring for the shift convention).
        num_frames: total frames to generate (including the n_prompt_frames seed frame(s)).
        decode_chunk_size: how many (batch*time)-flattened frames are decoded through the VAE per
            call at the end. The DDIM sampling loop above is memory-bounded regardless of
            num_frames (windowed to model.max_frames), but the final decode is NOT windowed --
            decoding B*num_frames frames in one call measurably OOMs a 40GB A100 once that
            product gets into the thousands (e.g. batch=4 x 600 frames). 64 is comfortably within
            what's already proven safe for a single ~600-frame episode's decode.

    Returns:
        (B, num_frames, H, W, 3) uint8 tensor, CPU.
    """
    x = seed_frames[:, None].to(device)  # (B, 1, 3, H, W)
    actions = actions.to(device)  # (B, T, 23)

    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    B = x.shape[0]
    H, W = x.shape[-2:]
    scaling_factor = 0.07843137255
    x = rearrange(x, "b t c h w -> (b t) c h w")
    with autocast("cuda", dtype=torch.bfloat16):
        x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
    x = x[:, :n_prompt_frames]

    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    iterator = range(n_prompt_frames, num_frames)
    if show_progress:
        iterator = tqdm(iterator, desc=desc)
    for i in iterator:
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr = x.clone()[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            with autocast("cuda", dtype=torch.bfloat16):
                v = model(x_curr, t, actions[:, start_frame : i + 1])

            x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    # Decode in chunks along the flattened (b*t) dim, NOT one call over all of it -- unlike the
    # DDIM loop above (bounded by the model's sliding attention window regardless of num_frames),
    # this decode has no such windowing: at batch>1 and num_frames=600, (b*t) can reach into the
    # thousands, and decoding that many frames through the VAE's own attention blocks in one call
    # measurably OOMs a 40GB A100 (batch=4 x 600 frames did, during this pipeline's own testing).
    # A single episode's decode chunk stays exactly as cheap as the original unbatched code's.
    #
    # Each decoded chunk is converted straight to compact uint8 and moved to CPU immediately,
    # NOT accumulated as float32 and only converted/moved at the end. Two distinct OOM bugs hit
    # this exact spot this session:
    #   1. GPU OOM: leaving all chunks on GPU before one big torch.cat -- at batch=8 x 600 frames
    #      (4800 total flattened frames), that concatenation alone tried to allocate ~12GB and
    #      OOM'd a 40GB A100, even though each individual vae.decode() call was already chunked
    #      and small.
    #   2. HOST RAM OOM: fixing #1 by moving chunks to CPU but keeping them as float32 -- 4800
    #      frames x ~2.76MB/frame (float32, 640x360x3) is ~13GB of HOST memory, which combined
    #      with the real video data + loaded models blew a 32GB SLURM --mem allocation (a
    #      cgroup-level kill, "OUT_OF_MEMORY" in sacct, not a Python exception -- the job dies
    #      silently with no traceback, only visible via `sacct`/the SLURM log's oom_kill line).
    # Converting to uint8 PER CHUNK before accumulating (not after concatenating everything)
    # cuts that accumulated memory 4x -- ~13GB float32 becomes ~3.3GB uint8, matching the size
    # of the real video data it's compared against instead of dwarfing it.
    decode_chunk = max(1, decode_chunk_size)
    decoded = []
    for i in range(0, x.shape[0], decode_chunk):
        chunk = vae.decode(x[i : i + decode_chunk] / scaling_factor)
        chunk = torch.clamp((chunk + 1) / 2, 0, 1)
        chunk = (chunk * 255).byte().cpu()
        decoded.append(chunk)
    x = torch.cat(decoded, dim=0)
    x = rearrange(x, "(b t) c h w -> b t h w c", t=num_frames)
    return x


@torch.no_grad()
def generate_rollout(
    model,
    vae,
    seed_frame: torch.Tensor,
    actions: torch.Tensor,
    num_frames: int,
    device: str,
    n_prompt_frames: int = 1,
    ddim_steps: int = 10,
    desc: str = "Generating",
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Single-episode entry point -- thin wrapper around generate_rollout_batch (batch size 1).
    Kept for every existing caller (finetune_oasis_generate.py, generate_luanti_split_grids.py,
    make_comparison_grid_oasis.py, etc.) so none of them need to change.

    Args:
        seed_frame: (3, H, W) float tensor in [0, 1].
        actions: (T, 23) float tensor, Craftium's native action vector, ALREADY shifted by one
            frame (actions[0] = zeros, actions[t] = the raw action that produced frame t) --
            matching finetune_oasis_data.py's convention. Caller shifts, not this function, so
            both training-data loading and generation share one shift implementation.
        num_frames: total frames to generate (including the n_prompt_frames seed frame(s)).

    Returns:
        (num_frames, H, W, 3) uint8 tensor, CPU.
    """
    return generate_rollout_batch(
        model, vae, seed_frame[None], actions[None], num_frames, device,
        n_prompt_frames=n_prompt_frames, ddim_steps=ddim_steps, desc=desc, show_progress=show_progress,
    )[0]


def shift_actions(raw_action: torch.Tensor) -> torch.Tensor:
    """raw_action: (T, 23) -> (T, 23) shifted by one frame (load_actions convention: actions[0]
    is unavailable/zero, actions[t] = raw_action[t-1] for t >= 1)."""
    return torch.cat([torch.zeros_like(raw_action[:1]), raw_action[:-1]], dim=0)
