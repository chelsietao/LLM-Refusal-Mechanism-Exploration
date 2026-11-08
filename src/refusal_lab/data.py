"""Prompt data loading and leakage-free train/validation/test splitting."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

SPLITS = ("direction_train", "layer_validation", "locked_test")


def load_pairs(path: str | Path) -> list[dict]:
    """Load paired prompts from a JSONL file with id/category/harmful/harmless keys."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"id", "category", "harmful", "harmless"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} missing keys {sorted(missing)}")
            rows.append(row)
    validate_pairs(rows)
    return rows


def validate_pairs(rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("no prompt pairs")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pair ids")
    texts = [r["harmful"] for r in rows] + [r["harmless"] for r in rows]
    if len(set(texts)) != len(texts):
        raise ValueError("duplicate prompt text across the dataset")
    for r in rows:
        if not r["harmful"].strip() or not r["harmless"].strip():
            raise ValueError(f"empty prompt in {r['id']}")


def split_pairs(
    rows: list[dict],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Stratified (by category) split. A harmful/harmless pair never crosses splits."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda r: r["id"])
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * fractions[1]))
        n_test = max(1, round(n * fractions[2]))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"category {cat!r} too small to split ({n} pairs)")
        out["direction_train"] += items[:n_train]
        out["layer_validation"] += items[n_train : n_train + n_val]
        out["locked_test"] += items[n_train + n_val :]
    for s in SPLITS:
        out[s].sort(key=lambda r: r["id"])
    return out
