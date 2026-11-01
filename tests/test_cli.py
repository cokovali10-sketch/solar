import pytest, subprocess, sys, json, os
from pathlib import Path

ENV = dict(os.environ)
ENV["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])


def test_cli_compress(tmp_path):
    text = "Please refactor `foo()` now. It must work."
    p = subprocess.run(
        [sys.executable, "-m", "solar.cli", "compress", "--json"],
        input=text, capture_output=True, text=True, cwd=str(tmp_path), env=ENV,
    )
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert "compressed" in data
    assert "foo()" in data["compressed"]


def test_cli_init(tmp_path):
    p = subprocess.run(
        [sys.executable, "-m", "solar.cli", "init", "--path", "solar.json"],
        capture_output=True, text=True, cwd=str(tmp_path), env=ENV,
    )
    assert p.returncode == 0
    assert (tmp_path / "solar.json").exists()


def test_cli_models(tmp_path):
    p = subprocess.run(
        [sys.executable, "-m", "solar.cli", "models"],
        capture_output=True, text=True, cwd=str(tmp_path), env=ENV,
    )
    assert p.returncode == 0
    assert "gpt-4o-mini" in p.stdout


def test_cli_config(tmp_path):
    p = subprocess.run(
        [sys.executable, "-m", "solar.cli", "config"],
        capture_output=True, text=True, cwd=str(tmp_path), env=ENV,
    )
    assert p.returncode == 0
    json.loads(p.stdout)
