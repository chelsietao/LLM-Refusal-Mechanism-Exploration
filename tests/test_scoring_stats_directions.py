import numpy as np
import torch

from refusal_lab.directions import difference_of_means, random_direction, shuffled_label_directions
from refusal_lab.scoring import is_refusal, safe_log_row
from refusal_lab.stats import bootstrap_ci, paired_bootstrap_diff


def test_scorer_ignores_topical_words_in_compliant_answers():
    assert not is_refusal("Sure! Note that this is illegal and harmful in many places, but here is how a kite works...")
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("I cannot provide instructions for that.")
    assert is_refusal("I can’t assist with this request.")


def test_safe_log_has_no_text():
    row = safe_log_row("some harmful text")
    assert set(row) == {"is_refusal", "response_sha256_16", "response_chars"}


def test_bootstrap():
    m, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1], seed=1)
    assert lo <= m <= hi
    m, lo, hi = paired_bootstrap_diff([1, 1, 1, 0], [0, 0, 1, 0], seed=1)
    assert m == 0.5 and lo <= m <= hi


def test_directions():
    torch.manual_seed(0)
    true = torch.zeros(8); true[0] = 1
    h = torch.randn(2, 200, 8) * 0.1 + 3 * true
    s = torch.randn(2, 200, 8) * 0.1
    raw, unit = difference_of_means(h, s)
    assert torch.allclose(unit.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert (unit @ true).min() > 0.99
    shuf_raw, shuf = shuffled_label_directions(h, s, seed=0)
    # Shuffled labels carry no class signal: the raw difference of means collapses to
    # sampling noise. (Its *orientation* is not random -- it follows the pooled high-variance
    # axis -- which is why the controls are norm-matched to the learned vector in steering.)
    assert shuf_raw.norm(dim=-1).max() < 0.1 * raw.norm(dim=-1).min()
    assert torch.allclose(shuf.norm(dim=-1), torch.ones(2), atol=1e-5)
    r = random_direction(8, seed=0)
    assert abs(r.norm().item() - 1) < 1e-5 and torch.equal(r, random_direction(8, seed=0))
