"""High-level optimizer combining compression, context, cache, routing, budget."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from .budget import BudgetManager
from .cache import Cache
from .compress import compress_messages, compress_text
from .config import SolarConfig
from .context import build_context, inject_context
from .models import Message, OptimizationReport, OptimizedRequest, Usage
from .protect import split
from .router import ModelRouter
from .storage import Storage
from .tokens import estimate_tokens, messages_tokens


class SolarOptimizer:
    def __init__(
        self,
        config: SolarConfig,
        storage: Storage | None = None,
        budget: BudgetManager | None = None,
        cache: Cache | None = None,
        router: ModelRouter | None = None,
        project_root: str | None = None,
    ):
        self.config = config
        self.storage = storage or Storage(config.db_path, project=config.project)
        self.budget = budget or BudgetManager(self.storage, config.budget)
        self.cache = cache or Cache(self.storage, config.cache)
        self.router = router or ModelRouter(config)
        self.project_root = project_root

    def close(self) -> None:
        self.storage.close()

    def _context_fingerprint(self, bundle) -> str:
        if bundle is None or not bundle.files:
            return ""
        digest = hashlib.sha256()
        for item in bundle.files:
            digest.update(item["path"].encode())
            digest.update(str(item.get("tokens", 0)).encode())
        digest.update(bundle.diff.encode()[:4096])
        return digest.hexdigest()

    def prepare(
        self,
        messages: list,
        model: str | None = None,
        project_root: str | None = None,
        use_context: bool = True,
        use_cache: bool = True,
        use_compression: bool = True,
        use_routing: bool | None = None,
    ) -> OptimizedRequest:
        report = OptimizationReport()
        root = project_root or self.project_root
        requested_model = model or self.config.model
        report.routed_model = requested_model

        # normalize messages
        normalized: list[Message] = []
        for item in messages:
            if isinstance(item, Message):
                normalized.append(Message(item.role, item.content))
            else:
                normalized.append(Message.from_dict(item))
        if not normalized:
            normalized.append(Message("user", ""))

        # original token count
        original_tokens = messages_tokens(normalized, requested_model)
        report.original_tokens = original_tokens

        # context
        context_bundle = None
        if use_context and self.config.context.enabled and root:
            last_user = next(
                (m.content for m in reversed(normalized) if m.role == "user"), ""
            )
            query = last_user or "project overview"
            context_bundle = build_context(root, query, self.config.context, requested_model)
            if context_bundle.text:
                normalized = inject_context(normalized, context_bundle)
                report.context_tokens = context_bundle.tokens
                report.context_files = len(context_bundle.files)
                report.notes.append(
                    f"context: {len(context_bundle.files)} files, {context_bundle.tokens} tokens"
                )

        fingerprint = self._context_fingerprint(context_bundle)

        # compression
        if use_compression:
            outcome = compress_messages(normalized, model=requested_model, config=self.config.compression)
            normalized = outcome.messages
            report.integrity_ok = outcome.integrity_ok
            report.notes.extend(outcome.notes)
            if not outcome.integrity_ok:
                report.notes.append("compression integrity failed; original protected spans may be lost")

        optimized_tokens = messages_tokens(normalized, requested_model)
        report.optimized_tokens = optimized_tokens
        report.protected_spans = sum(
            split(m.content).protected_count for m in normalized
        )

        # cache lookup
        cached_response = None
        if use_cache and self.cache.enabled:
            entry, score, kind = self.cache.lookup(
                normalized,
                model=requested_model,
                project=self.config.project,
                prompt_version=self.config.prompt_version,
                context_fingerprint=fingerprint,
            )
            if entry is not None:
                cached_response = entry.get("response")
                report.cache_hit = True
                report.cache_score = score
                report.notes.append(f"cache hit ({kind}, score {score:.3f})")

        # routing
        routing_enabled = self.config.router.enabled if use_routing is None else use_routing
        if routing_enabled:
            route = self.router.route(normalized, optimized_tokens, requested_model)
            if route.model and route.model != requested_model:
                report.routed_model = route.model
                report.routing_reason = route.reason
                # re-estimate tokens for new model (tokenization differs slightly)
                optimized_tokens = messages_tokens(normalized, route.model)
                report.optimized_tokens = optimized_tokens
            else:
                report.routed_model = requested_model
                report.routing_reason = route.reason
        else:
            report.routed_model = requested_model
            report.routing_reason = "routing disabled"

        report.finalize()
        return OptimizedRequest(
            messages=normalized,
            model=report.routed_model,
            report=report,
            cached_response=cached_response,
        )

    def record(
        self,
        report: OptimizationReport,
        usage: Usage | None = None,
        latency_ms: int = 0,
        status: str = "ok",
        task_kind: str = "",
        cache_key: str = "",
        verified: bool = False,
    ) -> None:
        usage = usage or Usage()
        self.storage.record_request(
            project=self.config.project,
            model=report.routed_model,
            original_tokens=report.original_tokens,
            optimized_tokens=report.optimized_tokens,
            saved_tokens=report.saved_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            cache_hit=1 if report.cache_hit else 0,
            cache_score=report.cache_score,
            latency_ms=latency_ms,
            status=status,
            task_kind=task_kind,
            routed_model=report.routed_model,
            routing_reason=report.routing_reason,
        )
        # store cache entry on a successful non-cached response? caller decides;
        # here we only touch if a cache_key was provided by caller.

    def store_cache(
        self,
        messages: list,
        model: str,
        response: dict[str, Any],
        context_fingerprint: str = "",
        verified: bool = False,
    ) -> str:
        return self.cache.store(
            messages=messages,
            model=model,
            project=self.config.project,
            prompt_version=self.config.prompt_version,
            response=response,
            context_fingerprint=context_fingerprint,
            verified=verified,
        )

    def process_response(self, text: str) -> str:
        if not self.config.compression.compress_assistant_history:
            return text
        result = compress_text(
            text,
            model=self.config.model,
            config=self.config.compression,
            role="assistant",
        )
        if not result.integrity_ok:
            return text
        return result.compressed

    def check_budget(self, estimated_tokens: int, estimated_cost: float = 0.0):
        return self.budget.check(estimated_tokens, estimated_cost)
