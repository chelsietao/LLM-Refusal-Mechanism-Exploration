import json
from collections import Counter
from pathlib import Path

import pytest

from refusal_lab.data import SPLITS, load_pairs, split_pairs, validate_pairs

DATA = Path(__file__).parent.parent / "data" / "prompts_v3.jsonl"


def test_dataset_size_and_categories():
    rows = load_pairs(DATA)
    assert len(rows) >= 200
    assert len({r["category"] for r in rows}) >= 10


def test_split_is_disjoint_complete_and_stratified():
    rows = load_pairs(DATA)
    sp = split_pairs(rows, seed=1)
    ids = [r["id"] for s in SPLITS for r in sp[s]]
    assert len(ids) == len(set(ids)) == len(rows)
    for cat in {r["category"] for r in rows}:
        for s in SPLITS:
            assert any(r["category"] == cat for r in sp[s])
    assert len(sp["locked_test"]) >= 40 and len(sp["layer_validation"]) >= 40


def test_split_deterministic():
    rows = load_pairs(DATA)
    assert split_pairs(rows, seed=3) == split_pairs(rows, seed=3)
    assert split_pairs(rows, seed=3) != split_pairs(rows, seed=4)


def test_validate_rejects_duplicates():
    r = {"id": "a", "category": "c", "harmful": "x", "harmless": "y"}
    with pytest.raises(ValueError):
        validate_pairs([r, dict(r, id="b")])
