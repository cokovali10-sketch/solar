"""Semantic cache for SOLAR."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any

from .config import CacheConfig
from .storage import Storage
from .tokens import estimate_tokens


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "this", "that", "it",
    "as", "at", "by", "from", "i", "you", "we", "they", "not", "no",
    "please", "just", "very", "really",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u0430-\u044f\u0410-\u042f\u0401\u0451]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _shingles(tokens: list[str], n: int = 3) -> list[str]:
    if len(tokens) < n:
        return list(tokens)
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class HashingEmbedder:
    """Deterministic hashed bag-of-words embedder (stdlib only)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _tokenize(text)
        features = tokens + _shingles(tokens, 3)
        for feature in features:
            # stable hash
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            weight = 1.0 + (0.5 if " " in feature else 0.0)
            vec[bucket] += sign * weight
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def _normalize_prompt(messages: list) -> str:
    parts: list[str] = []
    for message in messages:
        if hasattr(message, "role"):
            role = message.role
            content = message.content
        else:
            role = message.get("role", "user")
            content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        parts.append(f"{role}: {str(content).strip()}")
    return "\n".join(parts)


def _prompt_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, storage: Storage, config: CacheConfig, embedder: HashingEmbedder | None = None):
        self.storage = storage
        self.config = config
        self.embedder = embedder or HashingEmbedder()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled) and self.config.mode != "off"

    def normalize(self, messages: list) -> str:
        return _normalize_prompt(messages)

    def lookup(
        self,
        messages: list,
        model: str,
        project: str,
        prompt_version: str,
        context_fingerprint: str = "",
    ) -> tuple[dict[str, Any] | None, float, str]:
        if not self.enabled:
            return None, 0.0, ""

        normalized = _normalize_prompt(messages)
        key = _prompt_hash(normalized)
        now = time.time()

        # exact match first
        exact = self.storage.cache_get_exact(key)
        if exact is not None and self._fresh(exact, now) and self._fingerprint_ok(exact, context_fingerprint):
            self.storage.cache_touch(exact["cache_key"])
            return exact, 1.0, "exact"

        # semantic match
        if self.config.mode == "semantic" or self.config.mode == "safe":
            embedding = self.embedder.embed(normalized)
            entry, score = self.storage.cache_find_similar(
                project=project,
                model=model,
                prompt_version=prompt_version,
                embedding=embedding,
                threshold=self.config.similarity_threshold,
                now=now,
                limit=min(self.config.max_entries, 2000),
            )
            if entry is not None and self._fingerprint_ok(entry, context_fingerprint):
                # safe mode also requires verified flag
                if self.config.mode == "safe" and not entry.get("verified", False):
                    return None, score, ""
                self.storage.cache_touch(entry["cache_key"])
                return entry, score, "semantic"

        return None, 0.0, ""

    def store(
        self,
        messages: list,
        model: str,
        project: str,
        prompt_version: str,
        response: dict[str, Any],
        context_fingerprint: str = "",
        verified: bool = False,
    ) -> str:
        if not self.enabled:
            return ""
        normalized = _normalize_prompt(messages)
        key = _prompt_hash(normalized)
        embedding = self.embedder.embed(normalized)
        ttl = self.config.ttl_seconds if self.config.ttl_seconds > 0 else 0
        self.storage.cache_put(
            cache_key=key,
            project=project,
            model=model,
            prompt_version=prompt_version,
            normalized_prompt=normalized[:2000],
            content_fingerprint=context_fingerprint,
            response=response,
            embedding=embedding,
            verified=verified,
            ttl_seconds=ttl,
        )
        self.storage.cache_prune(self.config.max_entries, ttl)
        return key

    def mark_verified(self, cache_key: str, verified: bool = True) -> None:
        self.storage.cache_mark_verified(cache_key, verified)

    def clear(self, project: str | None = None) -> int:
        return self.storage.cache_clear(project)

    def list_entries(self, project: str | None = None, limit: int = 50) -> list[dict]:
        return self.storage.cache_list(project, limit)

    def _fresh(self, entry: dict[str, Any], now: float) -> bool:
        ttl = entry.get("ttl_seconds", 0) or 0
        if ttl <= 0:
            return True
        return (now - entry.get("created_at", now)) <= ttl

    def _fingerprint_ok(self, entry: dict[str, Any], current: str) -> bool:
        stored = entry.get("content_fingerprint", "") or ""
        if not current:
            return True
        if not stored:
            return True
        if self.config.mode == "safe":
            return stored == current
        return True


def estimate_cache_tokens(messages: list) -> int:
    return estimate_tokens(_normalize_prompt(messages), "")
