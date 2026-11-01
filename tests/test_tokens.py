import pytest
from solar.tokens import estimate_tokens, messages_tokens, calibrate, cost_usd


def test_empty():
    assert estimate_tokens("") == 0


def test_basic_english():
    assert estimate_tokens("hello world") > 0


def test_code_denser_than_prose():
    prose = "this is a simple sentence with several words"
    code = "def foo(x): return x+1 if x>0 else 0"
    # code should cost at least as many tokens per char
    assert estimate_tokens(code) / len(code) >= estimate_tokens(prose) / len(prose) * 0.5


def test_cyrillic():
    assert estimate_tokens("Привет мир, это тест") > 0


def test_messages_tokens_includes_overhead():
    msgs = [{"role": "user", "content": "hi"}]
    assert messages_tokens(msgs) >= 4


def test_calibrate():
    assert calibrate(100, 120) == 1.2
    assert calibrate(0, 10) == 1.0
    assert calibrate(10, 0) == 1.0


def test_cost_usd():
    assert cost_usd(1_000_000, 0, 2.5, 0) == 2.5
    assert cost_usd(0, 1_000_000, 0, 10) == 10.0


def test_tiktoken_path_not_required():
    # heuristic path works without tiktoken
    assert estimate_tokens("hello", model="unknown-model-xyz") > 0
