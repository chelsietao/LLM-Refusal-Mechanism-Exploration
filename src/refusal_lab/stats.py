"""Small statistics helpers (numpy only)."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_bootstrap: int = 10_000, confidence: float = 0.95, seed: int = 0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_bootstrap, len(v)), replace=True).mean(axis=1)
    a = 1.0 - confidence
    lo, hi = np.quantile(means, [a / 2, 1 - a / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a, b, n_bootstrap: int = 10_000, confidence: float = 0.95, seed: int = 0):
    """CI for mean(a) - mean(b) when a[i] and b[i] come from the same prompt."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci(a - b, n_bootstrap, confidence, seed)
