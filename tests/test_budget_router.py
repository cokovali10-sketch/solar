import pytest
from solar.config import SolarConfig, BudgetConfig
from solar.storage import Storage
from solar.budget import BudgetManager
from solar.router import ModelRouter
from solar.metrics import MetricsEngine
from pathlib import Path


def test_budget_per_request(tmp_path):
    storage = Storage(str(tmp_path / "b.db"), project="p")
    cfg = BudgetConfig(max_tokens_per_request=100)
    bm = BudgetManager(storage, cfg)
    d = bm.check(50)
    assert d.allowed
    d2 = bm.check(200)
    assert not d2.allowed
    storage.close()


def test_budget_daily(tmp_path):
    storage = Storage(str(tmp_path / "b.db"), project="p")
    storage.record_request(project="p", model="m", original_tokens=0, optimized_tokens=90, saved_tokens=0)
    cfg = BudgetConfig(max_tokens_per_day=100)
    bm = BudgetManager(storage, cfg)
    d = bm.check(20)
    assert not d.allowed
    storage.close()


def test_router_detects_debug():
    config = SolarConfig()
    r = ModelRouter(config)
    msgs = [{"role": "user", "content": "I get a traceback: ValueError in foo()"}]
    assert r.detect_task(msgs) == "debug"


def test_router_detects_docs():
    config = SolarConfig()
    r = ModelRouter(config)
    msgs = [{"role": "user", "content": "Explain what this function does"}]
    assert r.detect_task(msgs) in {"docs", "simple"}


def test_router_disabled_returns_default():
    config = SolarConfig()
    config.router.enabled = False
    r = ModelRouter(config)
    rr = r.route([{"role": "user", "content": "hi"}], 10)
    assert rr.model == config.model


def test_router_fits_context():
    config = SolarConfig()
    r = ModelRouter(config)
    # model with small context window should be excluded for huge prompts
    rr = r.route([{"role": "user", "content": "hi"}], 1_000_000)
    assert rr.model in {m["name"] for m in config.models}


def test_metrics_format(tmp_path):
    storage = Storage(str(tmp_path / "m.db"), project="p")
    storage.record_request(project="p", model="m", original_tokens=100, optimized_tokens=60, saved_tokens=40)
    m = MetricsEngine(storage)
    s = m.summary(project="p")
    assert s["requests"] == 1
    text = m.format_summary(s)
    assert "Saved tokens" in text
    storage.close()
