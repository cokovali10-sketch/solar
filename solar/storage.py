"""SQLite storage for SOLAR.

One database per installation (path from config). Schema is created on first use.
All writes go through a single lock so the gateway can be multi-threaded.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    project TEXT NOT NULL,
    model TEXT NOT NULL,
    original_tokens INTEGER NOT NULL,
    optimized_tokens INTEGER NOT NULL,
    saved_tokens INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    cache_score REAL NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ok',
    task_kind TEXT NOT NULL DEFAULT '',
    routed_model TEXT NOT NULL DEFAULT '',
    routing_reason TEXT NOT NULL DEFAULT '',
    context_fingerprint TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS cache_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT NOT NULL UNIQUE,
    project TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    normalized_prompt TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL DEFAULT '',
    response_json TEXT NOT NULL,
    embedding TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_used REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    verified INTEGER NOT NULL DEFAULT 0,
    ttl_seconds INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    started REAL NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_project_ts ON requests(project, ts);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
CREATE INDEX IF NOT EXISTS idx_cache_project_model ON cache_entries(project, model, prompt_version);
CREATE INDEX IF NOT EXISTS idx_cache_last_used ON cache_entries(last_used);
"""


class Storage:
    def __init__(self, db_path: str | Path, project: str = "default"):
        self.db_path = str(db_path)
        self.project = project
        self._lock = threading.RLock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- requests
    def record_request(self, **fields: Any) -> int:
        fields.setdefault("project", self.project)
        fields.setdefault("ts", time.time())
        columns = list(fields.keys())
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO requests ({', '.join(columns)}) VALUES ({placeholders})"
        with self._lock:
            cursor = self._conn.execute(sql, [fields[c] for c in columns])
            self._conn.commit()
            return int(cursor.lastrowid)

    def list_requests(self, limit: int = 50, project: str | None = None) -> list[dict]:
        project = project if project is not None else self.project
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM requests WHERE project = ? ORDER BY ts DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self, project: str | None = None, days: int = 30) -> dict[str, Any]:
        project = project if project is not None else self.project
        since = time.time() - days * 86400
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM requests WHERE project = ? AND ts >= ?",
                (project, since),
            ).fetchall()
        total = len(rows)
        original = sum(r["original_tokens"] for r in rows)
        optimized = sum(r["optimized_tokens"] for r in rows)
        saved = sum(r["saved_tokens"] for r in rows)
        prompt = sum(r["prompt_tokens"] for r in rows)
        completion = sum(r["completion_tokens"] for r in rows)
        cost = sum(r["cost_usd"] for r in rows)
        cache_hits = sum(1 for r in rows if r["cache_hit"])
        by_model: dict[str, dict[str, int]] = {}
        for r in rows:
            entry = by_model.setdefault(r["model"], {"requests": 0, "tokens": 0, "saved": 0})
            entry["requests"] += 1
            entry["tokens"] += r["optimized_tokens"]
            entry["saved"] += r["saved_tokens"]
        savings_pct = (100.0 * saved / original) if original else 0.0
        return {
            "project": project,
            "days": days,
            "requests": total,
            "original_tokens": original,
            "optimized_tokens": optimized,
            "saved_tokens": saved,
            "savings_pct": round(savings_pct, 2),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cost_usd": round(cost, 6),
            "cache_hits": cache_hits,
            "cache_hit_rate": round(100.0 * cache_hits / total, 2) if total else 0.0,
            "by_model": by_model,
        }

    def budget_totals(self, project: str | None = None, since: float | None = None) -> dict[str, int]:
        project = project if project is not None else self.project
        sql = "SELECT COALESCE(SUM(optimized_tokens),0) AS tokens, COALESCE(SUM(cost_usd),0) AS cost FROM requests WHERE project = ?"
        params: list[Any] = [project]
        if since is not None:
            sql += " AND ts >= ?"
            params.append(since)
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return {"tokens": int(row["tokens"]), "cost_usd": float(row["cost"])}

    # ------------------------------------------------------------------- cache
    def cache_get_exact(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cache_entries WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_cache(row)

    def cache_find_similar(
        self,
        project: str,
        model: str,
        prompt_version: str,
        embedding: list[float],
        threshold: float,
        now: float,
        limit: int = 500,
    ) -> tuple[dict[str, Any] | None, float]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM cache_entries
                   WHERE project = ? AND model = ? AND prompt_version = ?
                   AND (ttl_seconds = 0 OR ? - created_at <= ttl_seconds)
                   ORDER BY last_used DESC LIMIT ?""",
                (project, model, prompt_version, now, limit),
            ).fetchall()
        best: dict[str, Any] | None = None
        best_score = threshold
        for row in rows:
            entry = self._row_to_cache(row)
            stored = entry.get("_embedding", [])
            score = _cosine(embedding, stored)
            if score > best_score:
                best_score = score
                best = entry
        if best is None:
            return None, 0.0
        return best, best_score

    def cache_put(
        self,
        cache_key: str,
        project: str,
        model: str,
        prompt_version: str,
        normalized_prompt: str,
        content_fingerprint: str,
        response: dict[str, Any],
        embedding: list[float],
        verified: bool = False,
        ttl_seconds: int = 0,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO cache_entries
                   (cache_key, project, model, prompt_version, normalized_prompt,
                    content_fingerprint, response_json, embedding, created_at,
                    last_used, hits, verified, ttl_seconds)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                     response_json=excluded.response_json,
                     embedding=excluded.embedding,
                     content_fingerprint=excluded.content_fingerprint,
                     last_used=excluded.last_used,
                     verified=excluded.verified""",
                (
                    cache_key,
                    project,
                    model,
                    prompt_version,
                    normalized_prompt,
                    content_fingerprint,
                    json.dumps(response, ensure_ascii=False),
                    json.dumps(embedding),
                    now,
                    now,
                    0,
                    1 if verified else 0,
                    ttl_seconds,
                ),
            )
            self._conn.commit()

    def cache_touch(self, cache_key: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE cache_entries SET last_used = ?, hits = hits + 1 WHERE cache_key = ?",
                (time.time(), cache_key),
            )
            self._conn.commit()

    def cache_mark_verified(self, cache_key: str, verified: bool = True) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE cache_entries SET verified = ? WHERE cache_key = ?",
                (1 if verified else 0, cache_key),
            )
            self._conn.commit()

    def cache_prune(self, max_entries: int, ttl_seconds: int) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            if ttl_seconds > 0:
                cursor = self._conn.execute(
                    "DELETE FROM cache_entries WHERE ttl_seconds > 0 AND ? - created_at > ttl_seconds",
                    (now,),
                )
                removed += cursor.rowcount
            # LRU eviction
            count = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            if count > max_entries:
                excess = count - max_entries
                self._conn.execute(
                    """DELETE FROM cache_entries WHERE cache_key IN (
                        SELECT cache_key FROM cache_entries ORDER BY last_used ASC LIMIT ?
                    )""",
                    (excess,),
                )
                removed += excess
            self._conn.commit()
        return removed

    def cache_clear(self, project: str | None = None) -> int:
        with self._lock:
            if project is None:
                cursor = self._conn.execute("DELETE FROM cache_entries")
            else:
                cursor = self._conn.execute("DELETE FROM cache_entries WHERE project = ?", (project,))
            self._conn.commit()
            return cursor.rowcount

    def cache_list(self, project: str | None = None, limit: int = 50) -> list[dict]:
        project = project if project is not None else self.project
        with self._lock:
            rows = self._conn.execute(
                "SELECT cache_key, project, model, prompt_version, created_at, last_used, hits, verified, ttl_seconds, normalized_prompt FROM cache_entries WHERE project = ? ORDER BY last_used DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # ---------------------------------------------------------------------- kv
    def kv_get(self, key: str) -> tuple[Any, str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json, fingerprint FROM kv WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None, ""
        try:
            return json.loads(row["value_json"]), row["fingerprint"]
        except Exception:
            return None, row["fingerprint"]

    def kv_set(self, key: str, value: Any, fingerprint: str = "") -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO kv (key, value_json, fingerprint, updated_at) VALUES (?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
                   fingerprint=excluded.fingerprint, updated_at=excluded.updated_at""",
                (key, json.dumps(value, ensure_ascii=False), fingerprint, time.time()),
            )
            self._conn.commit()

    # ----------------------------------------------------------------- helpers
    def _row_to_cache(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            response = json.loads(row["response_json"])
        except Exception:
            response = {}
        try:
            embedding = json.loads(row["embedding"])
        except Exception:
            embedding = []
        return {
            "cache_key": row["cache_key"],
            "project": row["project"],
            "model": row["model"],
            "prompt_version": row["prompt_version"],
            "normalized_prompt": row["normalized_prompt"],
            "content_fingerprint": row["content_fingerprint"],
            "response": response,
            "created_at": row["created_at"],
            "last_used": row["last_used"],
            "hits": row["hits"],
            "verified": bool(row["verified"]),
            "ttl_seconds": row["ttl_seconds"],
            "_embedding": embedding,
        }


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))
