import torch
from accelerate.optimizer import AcceleratedOptimizer


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


class ChainedShuffleSampler(torch.utils.data.Sampler):
    """`num_passes` independent shuffles of range(n), back to back: epochs without epoch boundaries.

    accelerate re-synchronizes RNG across ranks (a collective) every time a prepared dataloader starts a new
    pass. If ranks reach the end of a pass on different steps, that collective pairs up with a different one
    on the other rank and the job hangs. Chaining the passes into one long sampler means the dataloader is
    iterated once, at the start, while all ranks are in step.
    """

    def __init__(self, n, num_passes, seed=0):
        self.n, self.num_passes, self.seed = n, num_passes, seed

    def __len__(self):
        return self.n * self.num_passes

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.num_passes):
            yield from torch.randperm(self.n, generator=g).tolist()


def train_loader_sampling_kwargs(dataset, cfg, world_size):
    """DataLoader kwargs for the train loader: plain shuffling, or (cfg.no_epoch_boundaries) one
    ChainedShuffleSampler long enough for the whole run on every rank."""
    if not getattr(cfg, "no_epoch_boundaries", False):
        return {"shuffle": True}
    samples_needed = cfg.total_steps * cfg.batch_size * world_size
    num_passes = samples_needed // len(dataset) + 2
    return {"sampler": ChainedShuffleSampler(len(dataset), num_passes, seed=cfg.random_seed)}


def compute_grad_norm(model):
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(grads) == 0:
        total_norm = 0.0
    else:
        norms = [torch.linalg.vector_norm(g, 2.0) for g in grads]
        total_norm = torch.linalg.vector_norm(torch.stack(norms), 2.0).item()
    return total_norm


def compute_step_and_weight_norm(optimizer, eps: float = 1e-10):
    step_norm_sq = 0.0
    weight_norm_sq = 0.0

    if isinstance(optimizer, AcceleratedOptimizer):
        unwrapped_optimizer = optimizer.optimizer
    else:
        unwrapped_optimizer = optimizer

    for group in optimizer.param_groups:
        lr = group["lr"]
        beta1, beta2 = group.get("betas", (0.9, 0.999))  # Adam defaults
        eps_opt = group.get("eps", 1e-8)  # for Adam-type

        for p in group["params"]:
            if p.grad is None:
                continue

            # ‖∇θ‖²
            g = p.grad.detach()

            # --- approximate update Δθ --------------------------------------
            if isinstance(unwrapped_optimizer, torch.optim.SGD):
                dstep = -lr * g  # momentum not included; add if you use it
            elif isinstance(unwrapped_optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                state = optimizer.state[p]
                # First and second moment estimates up to this point
                m = state.get("exp_avg", torch.zeros_like(p))
                v = state.get("exp_avg_sq", torch.zeros_like(p))
                denom = (v.sqrt() / (1 - beta2) ** 0.5) + eps_opt
                dstep = -lr * (m / (1 - beta1) + g) / denom  # 1-step look-ahead
            else:
                # Fall back to first-order estimate
                dstep = -lr * g

            step_norm_sq += torch.sum(dstep**2).item()
            weight_norm_sq += torch.sum(p.data**2).item()

    step_norm = step_norm_sq**0.5
    weight_norm = max(weight_norm_sq**0.5, eps)  # avoid /0

    return step_norm, weight_norm
