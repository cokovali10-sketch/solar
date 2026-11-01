import pytest
from pathlib import Path
from solar.config import SolarConfig
from solar.optimizer import SolarOptimizer


def test_prepare_basic():
    cfg = SolarConfig()
    cfg.db_path = ":memory:"
    opt = SolarOptimizer(cfg)
    msgs = [{"role": "user", "content": "Please refactor foo() to use bar(). It must not break tests."}]
    o = opt.prepare(msgs, use_context=False)
    assert o.report.original_tokens > 0
    assert o.report.integrity_ok
    assert "foo()" in " ".join(m.content for m in o.messages)
    opt.close()


def test_cache_hit_on_repeat(tmp_path):
    cfg = SolarConfig()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.cache.mode = "semantic"
    opt = SolarOptimizer(cfg)
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    o1 = opt.prepare(msgs, use_context=False)
    assert not o1.report.cache_hit
    opt.store_cache(o1.messages, o1.model, {"answer": "4"}, verified=True)
    o2 = opt.prepare(msgs, use_context=False)
    assert o2.report.cache_hit
    opt.close()


def test_budget_check(tmp_path):
    cfg = SolarConfig()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.budget.max_tokens_per_request = 5
    opt = SolarOptimizer(cfg)
    d = opt.check_budget(100)
    assert not d.allowed
    opt.close()


def test_context_injection(tmp_path):
    (tmp_path / "a.py").write_text("def target(): pass\n", encoding="utf-8")
    cfg = SolarConfig()
    cfg.db_path = str(tmp_path / "t.db")
    opt = SolarOptimizer(cfg, project_root=str(tmp_path))
    msgs = [{"role": "user", "content": "find target function"}]
    o = opt.prepare(msgs)
    assert o.report.context_files >= 1
    assert any("Project context" in m.content for m in o.messages)
    opt.close()
