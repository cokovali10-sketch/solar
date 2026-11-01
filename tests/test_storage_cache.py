import pytest
import tempfile, os
from solar.storage import Storage
from solar.cache import Cache, HashingEmbedder
from solar.config import CacheConfig


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test.db"), project="p")
    yield s
    s.close()


def test_record_and_stats(storage):
    storage.record_request(project="p", model="m", original_tokens=100, optimized_tokens=60, saved_tokens=40)
    stats = storage.stats(project="p", days=30)
    assert stats["requests"] == 1
    assert stats["saved_tokens"] == 40
    assert stats["savings_pct"] == 40.0


def test_budget_totals(storage):
    storage.record_request(project="p", model="m", original_tokens=10, optimized_tokens=10, saved_tokens=0)
    totals = storage.budget_totals("p")
    assert totals["tokens"] == 10


def test_cache_exact_roundtrip(storage):
    cache = Cache(storage, CacheConfig(mode="semantic"))
    msgs = [{"role": "user", "content": "hello world"}]
    key = cache.store(msgs, "m", "p", "v1", {"answer": 42})
    assert key
    entry, score, kind = cache.lookup(msgs, "m", "p", "v1")
    assert entry is not None
    assert kind == "exact"
    assert entry["response"]["answer"] == 42


def test_cache_miss_different(storage):
    cache = Cache(storage, CacheConfig(mode="semantic", similarity_threshold=0.99))
    cache.store([{"role": "user", "content": "hello world"}], "m", "p", "v1", {"a": 1})
    entry, score, kind = cache.lookup([{"role": "user", "content": "completely different topic about rust"}], "m", "p", "v1")
    assert kind != "exact"


def test_cache_safe_mode_fingerprint(storage):
    cache = Cache(storage, CacheConfig(mode="safe"))
    msgs = [{"role": "user", "content": "explain X"}]
    cache.store(msgs, "m", "p", "v1", {"a": 1}, context_fingerprint="fp1")
    e, s, k = cache.lookup(msgs, "m", "p", "v1", context_fingerprint="fp1")
    assert e is not None
    e2, s2, k2 = cache.lookup(msgs, "m", "p", "v1", context_fingerprint="fp2")
    assert e2 is None


def test_embedder_deterministic():
    e = HashingEmbedder()
    a = e.embed("hello world test")
    b = e.embed("hello world test")
    assert a == b


def test_cosine_identical():
    from solar.storage import _cosine
    v = [1.0, 2.0, 3.0]
    assert _cosine(v, v) > 0.99
