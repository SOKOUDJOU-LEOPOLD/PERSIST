"""Minimal LoRA (Hu et al., 2021) for fine-tuning: frozen base nn.Linear + trainable low-rank update.

    y = base(x) + (alpha / rank) * x @ A^T @ B^T,   A: (rank, in), B: (out, rank), B initialised to 0

so training starts exactly at the pretrained model. `merge_lora` folds the update back into plain nn.Linear
layers, giving a state dict with the original parameter names that the inference pipeline loads unchanged.
"""
import math

import torch
from torch import nn

DEFAULT_TARGETS = ("to_qkv", "to_q", "to_kv", "to_out", "fc1", "fc2")


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.scaling = rank, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.to(x.dtype).t() @ self.lora_B.to(x.dtype).t()) * self.scaling

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                        dtype=self.base.weight.dtype, device=self.base.weight.device)
        out.weight.copy_(self.base.weight + self.scaling * (self.lora_B @ self.lora_A))
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def inject_lora(model: nn.Module, rank: int, alpha: float | None = None, targets=DEFAULT_TARGETS) -> int:
    """Freeze every parameter of `model`, then wrap each nn.Linear whose attribute name is in `targets` with a
    trainable LoRALinear. Returns the number of wrapped layers."""
    alpha = rank if alpha is None else alpha
    model.requires_grad_(False)
    wrapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, rank, alpha))
                wrapped += 1
    return wrapped


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear in `model` with its merged nn.Linear, in place. Returns the number merged."""
    merged = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
                merged += 1
    return merged
