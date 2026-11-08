import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM, LlamaConfig, LlamaForCausalLM


def _llama(tie=True):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=120, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=tie)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def tiny_llama():
    return _llama(True)


@pytest.fixture
def tiny_llama_untied():
    return _llama(False)


@pytest.fixture
def tiny_gemma2():
    torch.manual_seed(0)
    cfg = Gemma2Config(vocab_size=120, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       query_pre_attn_scalar=8, sliding_window=16, max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    m = Gemma2ForCausalLM(cfg).eval()
    # random-init norm scales are ~0; make them non-trivial like trained weights
    for p in m.parameters():
        pass
    for layer in m.model.layers:
        for n in ("post_attention_layernorm", "post_feedforward_layernorm"):
            getattr(layer, n).weight.data.normal_(0, 0.5)
    return m


class FakeTokenizer:
    """Minimal tokenizer: chars -> ids, left padding, trivial chat template."""
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<u>" + messages[0]["content"] + "<m>"

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        ids = [[1 + (ord(c) % 100) for c in t][:40] for t in texts]
        w = max(map(len, ids))
        input_ids = torch.tensor([[0] * (w - len(x)) + x for x in ids])
        mask = torch.tensor([[0] * (w - len(x)) + [1] * len(x) for x in ids])
        return {"input_ids": input_ids, "attention_mask": mask}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(int(i)) for i in row if int(i) != 0) for row in ids]
