import pytest
from solar.compress import compress_text, compress_logs, compress_response


def test_protected_survives():
    text = "Please fix `load_config()` in D:\\SOLAR\\solar\\config.py. It must work."
    r = compress_text(text)
    assert r.integrity_ok
    assert "load_config()" in r.compressed
    assert "D:\\SOLAR\\solar\\config.py" in r.compressed


def test_constraint_sentence_kept():
    text = "Maybe refactor. It must not break tests. Thanks."
    r = compress_text(text, level="aggressive")
    assert "must not break tests" in r.compressed


def test_filler_removed():
    text = "Please could you kindly explain what this function does."
    r = compress_text(text, level="balanced")
    assert "please" not in r.compressed.lower() or "kindly" not in r.compressed.lower()


def test_phrase_map():
    text = "We need to do this in order to make sure that it works. Additionally, we utilize approximately 5 items."
    r = compress_text(text, level="balanced")
    assert "in order to" not in r.compressed.lower()


def test_min_savings_revert():
    text = "Short text."
    r = compress_text(text)
    assert r.integrity_ok


def test_log_compression():
    lines = ["2024-01-01 10:00:00 INFO starting"] * 10 + [
        "2024-01-01 10:00:01 ERROR boom happened",
        "2024-01-01 10:00:02 INFO done",
    ]
    text = "\n".join(lines)
    from solar.config import CompressionConfig
    out, note = compress_logs(text, CompressionConfig())
    assert note is not None
    assert "boom happened" in out
    assert len(out) < len(text)


def test_response_keeps_code():
    text = "Sure! Here is code.\n\n```python\ndef f(): return 1\n```\n\nHope this helps."
    r = compress_response(text)
    assert r.integrity_ok
    assert "def f" in r.compressed
    assert "```" in r.compressed


def test_integrity_failure_reverts():
    text = "Keep `IDENT` please."
    r = compress_text(text)
    assert r.integrity_ok
