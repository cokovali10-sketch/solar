"""Smart model router for SOLAR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import RouterConfig, SolarConfig


_TASK_KEYWORDS = {
    "debug": [
        "error", "exception", "traceback", "bug", "crash", "fails", "failing",
        "stack trace", "panic", "not working", "doesn't work", "broken",
        "ошибка", "баг", "падает", "не работает", "сломал",
    ],
    "architecture": [
        "architecture", "design", "refactor", "structure", "module", "system",
        "scalab", "pattern", "tradeoff", "trade-off", "plan",
        "архитектур", "дизайн", "рефактор", "структура", "модуль",
    ],
    "code": [
        "implement", "write code", "function", "class", "method", "api",
        "endpoint", "test", "unit test", "integration",
        "реализуй", "напиши код", "функция", "класс", "метод", "тест",
    ],
    "docs": [
        "explain", "what is", "how does", "document", "summarize", "describe",
        "explain", "объясни", "что такое", "как работает", "опиши", "документ",
    ],
}

_CODE_FENCE = re.compile(r"```")


@dataclass
class RouteResult:
    model: str
    task_kind: str
    reason: str


class ModelRouter:
    def __init__(self, config: SolarConfig):
        self.config = config

    def detect_task(self, messages: list) -> str:
        text = "\n".join(self._message_text(m) for m in messages).lower()
        scores = {kind: 0 for kind in _TASK_KEYWORDS}
        for kind, keywords in _TASK_KEYWORDS.items():
            for keyword in keywords:
                scores[kind] += text.count(keyword)
        # code fences and large diffs push toward code/debug
        if _CODE_FENCE.search(text):
            scores["code"] += 2
        if "traceback" in text or "exception" in text:
            scores["debug"] += 3
        # short prompts are simple
        if len(text) < 200 and not _CODE_FENCE.search(text):
            scores["simple"] = scores.get("simple", 0) + 3
        # pick highest; tie-break toward code over docs
        best_kind = "simple"
        best_score = -1
        for kind in ("simple", "docs", "code", "debug", "architecture"):
            score = scores.get(kind, 0)
            if score > best_score:
                best_score = score
                best_kind = kind
        if best_score == 0:
            # default by length
            if len(text) > 2000:
                best_kind = "architecture" if "design" in text else "code"
            else:
                best_kind = "simple"
        return best_kind

    def route(
        self,
        messages: list,
        estimated_tokens: int,
        requested_model: str | None = None,
    ) -> RouteResult:
        cfg = self.config.router
        if not cfg.enabled:
            model = requested_model or self.config.model or cfg.default_model
            return RouteResult(model=model, task_kind="default", reason="router disabled")

        task_kind = self.detect_task(messages)

        # user rules first
        text = "\n".join(self._message_text(m) for m in messages)
        for rule in cfg.rules or []:
            when = rule.get("when", "")
            target = rule.get("model", "")
            if when and target and re.search(when, text, re.IGNORECASE):
                if self._model_exists(target) and self._fits_context(target, estimated_tokens):
                    return RouteResult(target, task_kind, f"rule matched: {when}")

        required_quality = cfg.quality_by_task.get(task_kind, 3)
        candidates = []
        for model_info in self.config.models:
            name = model_info["name"]
            if not self._fits_context(name, estimated_tokens):
                continue
            if model_info.get("quality", 0) < required_quality:
                continue
            candidates.append(model_info)

        if not candidates:
            # relax quality requirement
            for model_info in self.config.models:
                name = model_info["name"]
                if self._fits_context(name, estimated_tokens):
                    candidates.append(model_info)

        if not candidates:
            model = requested_model or self.config.model or cfg.default_model
            return RouteResult(model, task_kind, "no fitting model; fallback")

        # prefer: quality first for hard tasks, cost first otherwise
        if task_kind in {"debug", "architecture"}:
            candidates.sort(key=lambda m: (-m.get("quality", 0), m.get("cost_in", 0)))
        else:
            candidates.sort(key=lambda m: (m.get("cost_in", 0), -m.get("quality", 0)))
        chosen = candidates[0]["name"]
        return RouteResult(chosen, task_kind, f"task={task_kind}, quality>={required_quality}")

    def model_info(self, name: str) -> dict[str, Any] | None:
        for info in self.config.models:
            if info["name"] == name:
                return info
        return None

    def cost_for(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        info = self.model_info(model)
        if not info:
            return 0.0
        from .tokens import cost_usd

        return cost_usd(
            prompt_tokens,
            completion_tokens,
            info.get("cost_in", 0.0),
            info.get("cost_out", 0.0),
        )

    def _model_exists(self, name: str) -> bool:
        return self.model_info(name) is not None

    def _fits_context(self, name: str, tokens: int) -> bool:
        info = self.model_info(name)
        if not info:
            return True
        return tokens <= info.get("context_window", 10**9)

    def _message_text(self, message) -> str:
        if hasattr(message, "content"):
            return str(message.content)
        return str(message.get("content", ""))
