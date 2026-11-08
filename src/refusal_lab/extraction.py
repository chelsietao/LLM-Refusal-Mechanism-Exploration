"""Prompt formatting, activation extraction and generation helpers."""
from __future__ import annotations

import torch

from .interventions import decoder_layers
from .scoring import safe_log_row


def format_prompts(tokenizer, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]


def _encode(tokenizer, texts, device):
    # The chat template already contains BOS, so do not add special tokens a second time.
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def last_token_residuals(model, tokenizer, formatted: list[str], batch_size: int = 8) -> torch.Tensor:
    """Residual stream after each decoder layer at the last prompt token: [layers, n, d].

    Captured with hooks on the decoder layers. ``output_hidden_states`` is avoided on purpose:
    its final entry is *after* the model's final norm, which would mix a different space into
    the last layer. Requires left padding so index -1 is the last real token.
    """
    layers = decoder_layers(model)
    per_layer: list[list[torch.Tensor]] = [[] for _ in layers]
    store: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_m, _in, out):
            t = out[0] if isinstance(out, tuple) else out
            store[i] = t[:, -1, :].detach().float().cpu()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        for s in range(0, len(formatted), batch_size):
            enc = _encode(tokenizer, formatted[s : s + batch_size], model.device)
            model(**enc, use_cache=False)
            for i in range(len(layers)):
                per_layer[i].append(store[i])
    finally:
        for h in handles:
            h.remove()
    return torch.stack([torch.cat(chunks, dim=0) for chunks in per_layer])


@torch.no_grad()
def generate(model, tokenizer, formatted: list[str], max_new_tokens: int = 64, batch_size: int = 8) -> list[str]:
    outs: list[str] = []
    for s in range(0, len(formatted), batch_size):
        enc = _encode(tokenizer, formatted[s : s + batch_size], model.device)
        width = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        outs += tokenizer.batch_decode(gen[:, width:], skip_special_tokens=True)
    return outs


def refusal_rate(responses: list[str]) -> float:
    return sum(safe_log_row(r)["is_refusal"] for r in responses) / max(1, len(responses))
