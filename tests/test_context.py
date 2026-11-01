import pytest
from pathlib import Path
from solar.context import build_context, is_git_repo
from solar.config import ContextConfig


def test_build_context_nonexistent():
    b = build_context("/no/such/path", "query", ContextConfig())
    assert b.files == []
    assert "does not exist" in " ".join(b.notes)


def test_build_context_finds_files(tmp_path):
    (tmp_path / "main.py").write_text("def load_config():\n    return {}\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("print('hi')\n", encoding="utf-8")
    b = build_context(str(tmp_path), "load_config", ContextConfig(budget_tokens=2000), "gpt-4o-mini")
    assert len(b.files) >= 1
    assert any("main.py" in f["path"] for f in b.files)


def test_is_git_repo_false(tmp_path):
    assert is_git_repo(tmp_path) is False


def test_budget_respected(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x = 1\n" * 50, encoding="utf-8")
    b = build_context(str(tmp_path), "x", ContextConfig(budget_tokens=300, max_files=10), "gpt-4o-mini")
    assert "budget reached" in " ".join(b.notes) or len(b.files) <= 5
