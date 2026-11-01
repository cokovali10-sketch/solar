import pytest
from solar.protect import split, integrity_check


def test_code_fence_protected():
    text = "Please fix\n```python\ndef foo(): pass\n```\nthanks"
    p = split(text)
    assert p.protected_count >= 1
    assert any("def foo" in t for t in p.protected_texts())


def test_inline_code_protected():
    p = split("Use `load_config()` here")
    assert any("load_config()" in t for t in p.protected_texts())


def test_path_protected():
    p = split("Edit D:\\SOLAR\\solar\\config.py now")
    assert any("config.py" in t for t in p.protected_texts())


def test_identifier_protected():
    p = split("Call loadConfig and SiteConfig")
    assert p.protected_count >= 2


def test_integrity_ok():
    text = "Fix `foo()` in /a/b.py"
    p = split(text)
    ok, missing = integrity_check(p, text)
    assert ok
    assert missing == []


def test_integrity_missing():
    text = "Fix `foo()` in /a/b.py"
    p = split(text)
    ok, missing = integrity_check(p, "Fix nothing")
    assert not ok
    assert missing


def test_roundtrip():
    text = "Hello `code()` world 123"
    p = split(text)
    assert p.text == text
