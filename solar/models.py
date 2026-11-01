"""Plain data structures shared across SOLAR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        content = data.get("content", "")
        if isinstance(content, list):
            # OpenAI content parts: keep text parts, ignore images for counting.
            content = "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return cls(role=str(data.get("role", "user")), content=str(content))


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    @classmethod
    def from_api(cls, data: dict[str, Any] | None) -> "Usage":
        if not data:
            return cls()
        prompt = int(data.get("prompt_tokens", 0) or 0)
        completion = int(data.get("completion_tokens", 0) or 0)
        total = int(data.get("total_tokens", prompt + completion) or 0)
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


@dataclass
class OptimizationReport:
    original_tokens: int = 0
    optimized_tokens: int = 0
    saved_tokens: int = 0
    savings_pct: float = 0.0
    protected_spans: int = 0
    integrity_ok: bool = True
    cache_hit: bool = False
    cache_score: float = 0.0
    routed_model: str = ""
    routing_reason: str = ""
    fallback_reason: str = ""
    context_tokens: int = 0
    context_files: int = 0
    notes: list[str] = field(default_factory=list)

    def finalize(self) -> "OptimizationReport":
        self.saved_tokens = max(0, self.original_tokens - self.optimized_tokens)
        if self.original_tokens > 0:
            self.savings_pct = round(100.0 * self.saved_tokens / self.original_tokens, 2)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "optimized_tokens": self.optimized_tokens,
            "saved_tokens": self.saved_tokens,
            "savings_pct": self.savings_pct,
            "protected_spans": self.protected_spans,
            "integrity_ok": self.integrity_ok,
            "cache_hit": self.cache_hit,
            "cache_score": round(self.cache_score, 4),
            "routed_model": self.routed_model,
            "routing_reason": self.routing_reason,
            "fallback_reason": self.fallback_reason,
            "context_tokens": self.context_tokens,
            "context_files": self.context_files,
            "notes": list(self.notes),
        }


@dataclass
class OptimizedRequest:
    messages: list[Message]
    model: str
    report: OptimizationReport
    cached_response: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [m.as_dict() for m in self.messages],
        }
