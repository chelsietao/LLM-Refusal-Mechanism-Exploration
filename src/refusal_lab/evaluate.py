"""End-to-end evaluation: all-layer ablation, steering and controls, with safe logging."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import pandas as pd
import torch

from .directions import difference_of_means, random_direction, shuffled_label_directions
from .extraction import format_prompts, generate, last_token_residuals
from .interventions import ablate_everywhere, ablate_layer, add_activation
from .scoring import safe_log_row
from .stats import bootstrap_ci


@dataclass
class EvalConfig:
    seed: int = 0
    batch_size: int = 8
    max_new_tokens: int = 64
    candidate_layers: list[int] | None = None  # None = every layer
    steer_coeffs: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass
class Directions:
    raw: torch.Tensor            # [L, d] unnormalised difference of means
    unit: torch.Tensor           # [L, d]
    shuffled_unit: torch.Tensor  # [L, d]
    random_unit: torch.Tensor    # [d]


def fit_directions(model, tokenizer, train_pairs: list[dict], cfg: EvalConfig) -> Directions:
    h = last_token_residuals(model, tokenizer, format_prompts(tokenizer, [p["harmful"] for p in train_pairs]), cfg.batch_size)
    s = last_token_residuals(model, tokenizer, format_prompts(tokenizer, [p["harmless"] for p in train_pairs]), cfg.batch_size)
    raw, unit = difference_of_means(h, s)
    _, shuf = shuffled_label_directions(h, s, cfg.seed)
    return Directions(raw, unit, shuf, random_direction(raw.shape[-1], cfg.seed))


def _responses(model, tokenizer, prompts, ctx, cfg):
    with ctx:
        return generate(model, tokenizer, format_prompts(tokenizer, prompts), cfg.max_new_tokens, cfg.batch_size)


def select_layer(model, tokenizer, val_pairs: list[dict], dirs: Directions, cfg: EvalConfig) -> pd.DataFrame:
    """Validation sweep: ablate the layer-l direction everywhere and measure harmful refusal."""
    harmful = [p["harmful"] for p in val_pairs]
    base = _responses(model, tokenizer, harmful, nullcontext(), cfg)
    base_rate = sum(safe_log_row(r)["is_refusal"] for r in base) / len(base)
    layers = cfg.candidate_layers if cfg.candidate_layers is not None else list(range(dirs.unit.shape[0]))
    rows = []
    for l in layers:
        resp = _responses(model, tokenizer, harmful, ablate_everywhere(model, dirs.unit[l]), cfg)
        rate = sum(safe_log_row(r)["is_refusal"] for r in resp) / len(resp)
        rows.append({"layer": l, "baseline_refusal": base_rate, "ablated_refusal": rate,
                     "direction_norm": float(dirs.raw[l].norm())})
    return pd.DataFrame(rows).sort_values(["ablated_refusal", "layer"]).reset_index(drop=True)


def _log(responses, pairs, condition, intent, extra=None):
    rows = []
    for r, p in zip(responses, pairs):
        row = {"condition": condition, "intent": intent, "pair_id": p["id"],
               "category": p["category"], **safe_log_row(r)}
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def evaluate_locked_test(model, tokenizer, test_pairs: list[dict], dirs: Directions, layer: int, cfg: EvalConfig) -> pd.DataFrame:
    """Per-example log for every condition on the locked test split."""
    harmful = [p["harmful"] for p in test_pairs]
    harmless = [p["harmless"] for p in test_pairs]
    rows: list[dict] = []

    ablations = {
        "baseline": lambda: nullcontext(),
        "ablate_all_learned": lambda: ablate_everywhere(model, dirs.unit[layer]),
        "ablate_all_shuffled": lambda: ablate_everywhere(model, dirs.shuffled_unit[layer]),
        "ablate_all_random": lambda: ablate_everywhere(model, dirs.random_unit),
        "ablate_single_layer_learned": lambda: ablate_layer(model, layer, dirs.unit[layer]),
    }
    for name, make in ablations.items():
        rows += _log(_responses(model, tokenizer, harmful, make(), cfg), test_pairs, name, "harmful")
        rows += _log(_responses(model, tokenizer, harmless, make(), cfg), test_pairs, name, "harmless")

    # Steering on harmless prompts; norm-matched to the learned vector at this layer.
    norm = float(dirs.raw[layer].norm())
    for c in cfg.steer_coeffs:
        for tag, unit in (("learned", dirs.unit[layer]), ("shuffled", dirs.shuffled_unit[layer]), ("random", dirs.random_unit)):
            vec = c * norm * unit
            resp = _responses(model, tokenizer, harmless, add_activation(model, layer, vec), cfg)
            rows += _log(resp, test_pairs, f"steer_{tag}", "harmless", {"coeff": c})
    return pd.DataFrame(rows)


def summarize(log: pd.DataFrame, n_bootstrap: int = 10_000, seed: int = 0) -> pd.DataFrame:
    keys = [k for k in ("condition", "intent", "coeff") if k in log.columns]
    out = []
    for key, g in log.groupby(keys, dropna=False):
        mean, lo, hi = bootstrap_ci(g["is_refusal"].astype(float).to_numpy(), n_bootstrap, seed=seed)
        out.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                    "refusal_rate": mean, "ci95_low": lo, "ci95_high": hi, "n": len(g)})
    return pd.DataFrame(out)
