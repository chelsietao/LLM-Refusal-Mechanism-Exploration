"""Bake directional ablation into the weights (zero runtime overhead).

For every matrix that writes into the residual stream we replace W with (I - r r^T) W:
token embeddings (rows), each attention ``o_proj`` and each MLP ``down_proj``.

This is *exactly* equivalent to the all-position hook ablation only when nothing
non-linear/element-wise sits between those matrices and the residual add. Gemma-2/3 apply
an RMSNorm (with a learned per-channel scale) to the attention/MLP outputs first, so for
those models the edit is only approximate and this module refuses unless asked to
(``allow_inexact=True``).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .interventions import decoder_layers, has_post_sublayer_norms, _base


def supports_exact_orthogonalization(model) -> bool:
    return not has_post_sublayer_norms(model)


def _project_rows(weight: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    # embedding [V, d_model]: remove the component of each row along d
    w = weight.float()
    return (w - (w @ d).unsqueeze(-1) * d).to(weight.dtype)


def _project_cols_out(weight: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    # linear weight [d_model, in]: (I - d d^T) W
    w = weight.float()
    return (w - torch.outer(d, d @ w)).to(weight.dtype)


def _targets(model):
    base = _base(model)
    out = [("embed", base.embed_tokens)]
    for i, layer in enumerate(base.layers):
        out.append((f"layer{i}.o_proj", layer.self_attn.o_proj))
        out.append((f"layer{i}.down_proj", layer.mlp.down_proj))
    return out


@contextmanager
def orthogonalized(model, direction: torch.Tensor, allow_inexact: bool = False):
    """Temporarily edit weights in place; originals are restored on exit."""
    if not supports_exact_orthogonalization(model) and not allow_inexact:
        raise ValueError(
            "This architecture applies a norm to attention/MLP outputs before the residual "
            "add, so weight orthogonalization is not equivalent to hook ablation. "
            "Pass allow_inexact=True to apply it anyway."
        )
    saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    tied = None
    lm_head = getattr(model, "lm_head", None)
    base_embed = _base(model).embed_tokens
    if lm_head is not None and lm_head.weight is base_embed.weight:
        # Editing a tied embedding would also edit the output head; keep the head intact.
        tied = lm_head.weight
        lm_head.weight = torch.nn.Parameter(tied.detach().clone(), requires_grad=False)
    try:
        d_cache: dict[torch.device, torch.Tensor] = {}
        for name, module in _targets(model):
            w = module.weight
            d = d_cache.setdefault(w.device, direction.to(device=w.device, dtype=torch.float32))
            saved.append((w, w.detach().clone()))
            new = _project_rows(w.data, d) if name == "embed" else _project_cols_out(w.data, d)
            w.data.copy_(new)
            b = getattr(module, "bias", None)
            if b is not None:  # bias of a residual writer must also lose its r component
                saved.append((b, b.detach().clone()))
                bf = b.data.float()
                b.data.copy_((bf - (bf @ d) * d).to(b.dtype))
        yield
    finally:
        for p, original in saved:
            p.data.copy_(original)
        if tied is not None:
            lm_head.weight = tied
