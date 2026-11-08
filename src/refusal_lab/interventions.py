"""Forward-hook interventions on the residual stream.

* ``ablate_everywhere``  – project the direction out of *every* write into the residual
  stream (token embedding, every attention output, every MLP output) at *all* token
  positions, as in Arditi et al. (2024).
* ``ablate_layer``       – project it out of a single decoder layer's output (v1/v2 behaviour).
* ``add_activation``     – add a vector to one decoder layer's output at all positions (steering).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def _base(model):
    return model.model if hasattr(model, "model") else model


def decoder_layers(model):
    return list(_base(model).layers)


def has_post_sublayer_norms(model) -> bool:
    """Gemma-2/3 style: a norm is applied to attention/MLP outputs *before* the residual add."""
    layers = decoder_layers(model)
    return bool(layers) and hasattr(layers[0], "post_feedforward_layernorm")


def residual_writers(model):
    """Modules whose output is added to the residual stream."""
    base = _base(model)
    writers = [base.embed_tokens]
    post = has_post_sublayer_norms(model)
    for layer in base.layers:
        writers.append(layer.post_attention_layernorm if post else layer.self_attn)
        writers.append(layer.post_feedforward_layernorm if post else layer.mlp)
    return writers


def project_out(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    d = direction.to(device=x.device, dtype=torch.float32)
    xf = x.float()
    return (xf - (xf @ d).unsqueeze(-1) * d).to(x.dtype)


def _map_first(output, fn):
    if isinstance(output, tuple):
        return (fn(output[0]),) + tuple(output[1:])
    return fn(output)


@contextmanager
def _registered(pairs):
    handles = [module.register_forward_hook(hook) for module, hook in pairs]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def ablate_everywhere(model, direction: torch.Tensor):
    def hook(_m, _i, out):
        return _map_first(out, lambda t: project_out(t, direction))

    with _registered([(m, hook) for m in residual_writers(model)]):
        yield


@contextmanager
def ablate_layer(model, layer_idx: int, direction: torch.Tensor):
    layers = decoder_layers(model)
    if not 0 <= layer_idx < len(layers):
        raise IndexError(f"layer {layer_idx} outside [0, {len(layers) - 1}]")

    def hook(_m, _i, out):
        return _map_first(out, lambda t: project_out(t, direction))

    with _registered([(layers[layer_idx], hook)]):
        yield


@contextmanager
def add_activation(model, layer_idx: int, vector: torch.Tensor):
    """Steering: h <- h + vector at one layer's output, all positions."""
    layers = decoder_layers(model)
    if not 0 <= layer_idx < len(layers):
        raise IndexError(f"layer {layer_idx} outside [0, {len(layers) - 1}]")

    def hook(_m, _i, out):
        return _map_first(out, lambda t: t + vector.to(device=t.device, dtype=t.dtype))

    with _registered([(layers[layer_idx], hook)]):
        yield
