"""MLX server pieces that do not need the GPU: request shaping and the JSON-schema token mask.

The mask test uses the real Qwen tokenizer from the local Hugging Face cache and is
skipped when it (or mlx / llguidance, Apple Silicon only) is not installed.
"""

import json

import pytest

mlx_app = pytest.importorskip("mlx_server.app", reason="mlx-lm not installed (Apple Silicon only)")
engine_mod = pytest.importorskip("mlx_server.engine")

MODEL = engine_mod.MODEL_ID


def test_system_messages_are_merged_first():
    msgs = mlx_app._messages([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "developer", "content": "schema"},
        {"role": "tool", "content": "retry: bad json"},
    ])
    assert msgs[0] == {"role": "system", "content": "rules\n\nschema"}
    assert [m["role"] for m in msgs] == ["system", "user", "user"]
    assert msgs[1]["content"] == "hi"


def test_response_format_to_schema():
    s = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert mlx_app._schema({"type": "json_schema", "json_schema": {"name": "x", "schema": s}}) == s
    assert mlx_app._schema({"type": "json_object"}) == {"type": "object"}
    assert mlx_app._schema({"type": "text"}) is None
    assert mlx_app._schema(None) is None


@pytest.fixture(scope="module")
def tokenizers():
    transformers = pytest.importorskip("transformers")
    hf = pytest.importorskip("llguidance.hf")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} not in the local HF cache")
    vocab = len(tok) + 64  # logits are usually padded past the tokenizer vocab
    return tok, hf.from_tokenizer(tok, n_vocab=vocab, eos_token=tok.convert_tokens_to_ids("<|im_end|>")), vocab


def test_constraint_allows_valid_json_and_blocks_the_rest(tokenizers):
    mx = pytest.importorskip("mlx.core")
    from harness.schemas import MonitorVerdict

    tok, ll, vocab = tokenizers
    c = engine_mod.JsonConstraint(ll, MonitorVerdict.model_json_schema(), vocab)
    answer = json.dumps({"verdict": "stop", "evidence_epoch": 3, "evidence": "loss nan", "reason": "dead"},
                        separators=(",", ":"))
    ids = tok.encode(answer, add_special_tokens=False)
    seen = mx.array([1, 2, 3])  # prompt tail
    logits = mx.zeros((1, vocab))
    for t in ids:
        out = c(seen, logits)
        assert out[0, t].item() == 0.0, f"valid token {tok.decode([t])!r} was masked"
        seen = mx.concatenate([seen, mx.array([t])])
    # after a complete object only end-of-turn may follow
    out = c(seen, logits)
    allowed = [i for i in range(vocab) if out[0, i].item() == 0.0]
    assert allowed == [tok.convert_tokens_to_ids("<|im_end|>")]
    assert c.failed is None


def test_constraint_rejects_out_of_enum_value(tokenizers):
    mx = pytest.importorskip("mlx.core")
    from harness.schemas import MonitorVerdict

    tok, ll, vocab = tokenizers
    c = engine_mod.JsonConstraint(ll, MonitorVerdict.model_json_schema(), vocab)
    seen = mx.array([1])
    for t in tok.encode('{"verdict":"', add_special_tokens=False):
        c(seen, mx.zeros((1, vocab)))
        seen = mx.concatenate([seen, mx.array([t])])
    out = c(seen, mx.zeros((1, vocab)))
    bad = tok.encode("maybe", add_special_tokens=False)[0]
    good = tok.encode("continue", add_special_tokens=False)[0]
    assert out[0, bad].item() == float("-inf")
    assert out[0, good].item() == 0.0
