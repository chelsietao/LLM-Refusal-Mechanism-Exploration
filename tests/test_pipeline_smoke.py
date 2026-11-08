import torch

from conftest import FakeTokenizer
from refusal_lab.bench import compare_logits, time_condition
from refusal_lab.evaluate import EvalConfig, evaluate_locked_test, fit_directions, select_layer, summarize
from refusal_lab.extraction import format_prompts, last_token_residuals
from contextlib import nullcontext


def _pairs(n, prefix):
    return [{"id": f"{prefix}{i}", "category": "c", "harmful": f"harmful request number {i}", "harmless": f"harmless request number {i}"} for i in range(n)]


def test_last_token_residuals_matches_layer_outputs_and_ignores_padding(tiny_llama):
    tok = FakeTokenizer()
    texts = format_prompts(tok, ["short", "a much longer prompt here"])
    acts = last_token_residuals(tiny_llama, tok, texts, batch_size=2)
    assert acts.shape == (3, 2, 32)
    # batch composition (hence padding) must not change the activation of a prompt
    solo = last_token_residuals(tiny_llama, tok, texts[:1], batch_size=1)
    assert torch.allclose(acts[:, 0], solo[:, 0], atol=1e-4)


def test_full_pipeline_runs_and_logs_no_text(tiny_llama):
    tok = FakeTokenizer()
    cfg = EvalConfig(batch_size=4, max_new_tokens=4, steer_coeffs=(1.0,), candidate_layers=[0, 1, 2])
    dirs = fit_directions(tiny_llama, tok, _pairs(8, "t"), cfg)
    assert dirs.unit.shape == (3, 32) and dirs.shuffled_unit.shape == (3, 32)
    sweep = select_layer(tiny_llama, tok, _pairs(4, "v"), dirs, cfg)
    assert set(sweep["layer"]) == {0, 1, 2}
    log = evaluate_locked_test(tiny_llama, tok, _pairs(4, "x"), dirs, int(sweep.iloc[0]["layer"]), cfg)
    assert {"baseline", "ablate_all_learned", "ablate_all_shuffled", "ablate_all_random",
            "ablate_single_layer_learned", "steer_learned", "steer_shuffled", "steer_random"} <= set(log["condition"])
    assert not any("harmful request" in str(v) for v in log.to_numpy().ravel())
    summ = summarize(log, n_bootstrap=200)
    assert (summ["ci95_low"] <= summ["refusal_rate"]).all()


def test_bench_and_fidelity_run_on_cpu(tiny_llama):
    tok = FakeTokenizer()
    prompts = [f"prompt {i}" for i in range(4)]
    r = time_condition(tiny_llama, tok, prompts, nullcontext, new_tokens=4, batch_size=4, warmup=1, repeats=4)
    assert r["median_total_s"] > 0 and r["tokens_per_s"] > 0
    d = torch.nn.functional.normalize(torch.randn(32, generator=torch.Generator().manual_seed(0)), dim=0)
    fid = compare_logits(tiny_llama, tok, prompts, d, batch_size=4)
    assert fid["max_abs_logprob_diff"] < 1e-3 and fid["argmax_agreement"] == 1.0
