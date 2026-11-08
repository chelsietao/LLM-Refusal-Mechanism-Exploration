"""Inference benchmarks: hook ablation vs. weight-orthogonalized model (and fidelity check)."""
from __future__ import annotations

import statistics
import time
from contextlib import nullcontext

import torch

from .extraction import _encode, format_prompts
from .interventions import ablate_everywhere
from .orthogonalize import orthogonalized


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def _timed_generate(model, enc, new_tokens, pad_id):
    _sync(model.device)
    t0 = time.perf_counter()
    model.generate(**enc, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=pad_id)
    _sync(model.device)
    return time.perf_counter() - t0


def time_condition(model, tokenizer, prompts, ctx_factory, new_tokens=64, batch_size=8, warmup=2, repeats=7):
    """Median/IQR wall time for a fixed number of generated tokens.

    Prefill is measured separately (1 new token); decode ms/token is the remainder.
    """
    enc = _encode(tokenizer, format_prompts(tokenizer, prompts[:batch_size]), model.device)
    pad = tokenizer.pad_token_id
    with ctx_factory():
        for _ in range(warmup):
            _timed_generate(model, enc, new_tokens, pad)
        if str(model.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        totals = [_timed_generate(model, enc, new_tokens, pad) for _ in range(repeats)]
        prefills = [_timed_generate(model, enc, 1, pad) for _ in range(repeats)]
        peak = torch.cuda.max_memory_allocated() / 2**20 if str(model.device).startswith("cuda") else float("nan")
    q = statistics.quantiles(totals, n=4) if len(totals) >= 4 else [min(totals), statistics.median(totals), max(totals)]
    med_total, med_prefill = statistics.median(totals), statistics.median(prefills)
    return {
        "median_total_s": med_total,
        "iqr_total_s": q[2] - q[0],
        "median_prefill_s": med_prefill,
        "decode_ms_per_token": 1000 * (med_total - med_prefill) / max(1, new_tokens - 1),
        "tokens_per_s": batch_size * new_tokens / med_total,
        "peak_mem_mib": peak,
        "repeats": repeats,
        "batch_size": batch_size,
        "new_tokens": new_tokens,
    }


def benchmark_all(model, tokenizer, prompts, direction, **kw):
    """baseline vs hook ablation vs weight orthogonalization on the same inputs."""
    return {
        "baseline": time_condition(model, tokenizer, prompts, nullcontext, **kw),
        "hook_ablation": time_condition(model, tokenizer, prompts, lambda: ablate_everywhere(model, direction), **kw),
        "weight_orthogonalized": time_condition(model, tokenizer, prompts, lambda: orthogonalized(model, direction), **kw),
    }


@torch.no_grad()
def compare_logits(model, tokenizer, prompts, direction, batch_size=8):
    """How close is the weight-edited model to the hook-ablated model? (next-token log-probs)"""
    formatted = format_prompts(tokenizer, prompts)

    def collect():
        outs = []
        for s in range(0, len(formatted), batch_size):
            enc = _encode(tokenizer, formatted[s : s + batch_size], model.device)
            outs.append(torch.log_softmax(model(**enc, use_cache=False).logits[:, -1, :].float(), dim=-1).cpu())
        return torch.cat(outs)

    with ablate_everywhere(model, direction):
        a = collect()
    with orthogonalized(model, direction):
        b = collect()
    kl = (a.exp() * (a - b)).sum(-1)
    return {
        "max_abs_logprob_diff": float((a - b).abs().max()),
        "mean_kl_hook_to_weights": float(kl.mean()),
        "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
    }
