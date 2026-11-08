import pytest
import torch
import torch.nn.functional as F

from refusal_lab.interventions import (ablate_everywhere, ablate_layer, add_activation,
                                       residual_writers, has_post_sublayer_norms)
from refusal_lab.orthogonalize import orthogonalized, supports_exact_orthogonalization


def _inputs():
    g = torch.Generator().manual_seed(1)
    return torch.randint(1, 120, (3, 11), generator=g)


def _direction(d=32, seed=2):
    return F.normalize(torch.randn(d, generator=torch.Generator().manual_seed(seed)), dim=0)


def test_writers_count_and_gemma_detection(tiny_llama, tiny_gemma2):
    assert len(residual_writers(tiny_llama)) == 1 + 2 * 3
    assert not has_post_sublayer_norms(tiny_llama) and has_post_sublayer_norms(tiny_gemma2)


def test_ablate_everywhere_removes_component_at_every_write(tiny_llama):
    d = _direction()
    # the residual stream must carry no component along d under ablation
    resid = []
    hooks = [l.register_forward_hook(lambda _m, _i, o: resid.append((o[0] if isinstance(o, tuple) else o) @ d))
             for l in tiny_llama.model.layers]
    with torch.no_grad(), ablate_everywhere(tiny_llama, d):
        tiny_llama(_inputs())
    for h in hooks:
        h.remove()
    assert all(r.abs().max() < 1e-4 for r in resid)
    # control: without ablation the component is clearly non-zero
    resid.clear()
    hooks = [l.register_forward_hook(lambda _m, _i, o: resid.append((o[0] if isinstance(o, tuple) else o) @ d))
             for l in tiny_llama.model.layers]
    with torch.no_grad():
        tiny_llama(_inputs())
    for h in hooks:
        h.remove()
    assert max(r.abs().max() for r in resid) > 1e-2


def test_hooks_are_removed(tiny_llama):
    d = _direction()
    x = _inputs()
    with torch.no_grad():
        before = tiny_llama(x).logits
        with ablate_everywhere(tiny_llama, d):
            changed = tiny_llama(x).logits
        with ablate_layer(tiny_llama, 1, d):
            pass
        with add_activation(tiny_llama, 1, d):
            pass
        after = tiny_llama(x).logits
    assert not torch.allclose(before, changed, atol=1e-4)
    assert torch.equal(before, after)


def test_single_layer_ablation_differs_from_everywhere(tiny_llama):
    d = _direction(); x = _inputs()
    with torch.no_grad():
        with ablate_everywhere(tiny_llama, d):
            a = tiny_llama(x).logits
        with ablate_layer(tiny_llama, 1, d):
            b = tiny_llama(x).logits
    assert not torch.allclose(a, b, atol=1e-4)


def test_add_activation_shifts_layer_output(tiny_llama):
    d = _direction(); x = _inputs()
    with torch.no_grad():
        base = tiny_llama(x).logits
        with add_activation(tiny_llama, 1, 5.0 * d):
            steered = tiny_llama(x).logits
    assert not torch.allclose(base, steered, atol=1e-4)
    with pytest.raises(IndexError):
        with add_activation(tiny_llama, 99, d):
            pass


@pytest.mark.parametrize("fixture", ["tiny_llama", "tiny_llama_untied"])
def test_weight_orthogonalization_equals_hook_ablation_on_llama(request, fixture):
    model = request.getfixturevalue(fixture)
    assert supports_exact_orthogonalization(model)
    d = _direction(); x = _inputs()
    originals = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        with ablate_everywhere(model, d):
            hook_logits = model(x).logits
        with orthogonalized(model, d):
            weight_logits = model(x).logits
    assert torch.allclose(hook_logits, weight_logits, atol=1e-4), (hook_logits - weight_logits).abs().max()
    # weights fully restored, tie preserved
    for k, v in model.state_dict().items():
        assert torch.equal(v, originals[k]), k
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == (fixture == "tiny_llama")


def test_orthogonalization_is_not_exact_on_gemma2_and_refuses(tiny_gemma2):
    m = tiny_gemma2
    assert not supports_exact_orthogonalization(m)
    d = _direction(); x = _inputs()
    with pytest.raises(ValueError):
        with orthogonalized(m, d):
            pass
    with torch.no_grad():
        with ablate_everywhere(m, d):
            hook_logits = m(x).logits
        with orthogonalized(m, d, allow_inexact=True):
            weight_logits = m(x).logits
    # The post-sublayer RMSNorm (rms + per-channel scale) breaks the equivalence.
    assert (hook_logits - weight_logits).abs().max() > 1e-3
