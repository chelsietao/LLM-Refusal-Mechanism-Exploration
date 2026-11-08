"""Direction estimators: difference of means, shuffled-label control, random control."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def difference_of_means(harmful: torch.Tensor, harmless: torch.Tensor):
    """harmful/harmless: [layers, n, d]. Returns (raw [layers, d], unit [layers, d])."""
    raw = harmful.float().mean(dim=1) - harmless.float().mean(dim=1)
    return raw, F.normalize(raw, dim=-1)


def shuffled_label_directions(harmful: torch.Tensor, harmless: torch.Tensor, seed: int = 0):
    """Pool all activations, randomly reassign the harmful/harmless labels
    (keeping group sizes), then take the difference of means.
    Control for 'any split of the data yields a direction'."""
    pooled = torch.cat([harmful, harmless], dim=1).float()
    n_h = harmful.shape[1]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(pooled.shape[1], generator=g)
    raw = pooled[:, perm[:n_h]].mean(dim=1) - pooled[:, perm[n_h:]].mean(dim=1)
    return raw, F.normalize(raw, dim=-1)


def random_direction(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(dim, generator=g), dim=0)
