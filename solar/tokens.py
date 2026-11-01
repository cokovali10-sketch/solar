"""Token estimation.

Exact counting requires the provider tokenizer. SOLAR uses `tiktoken` when it is
installed and falls back to a character-class heuristic otherwise. The heuristic
is intentionally conservative (it over-estimates slightly) so budget checks fail
early rather than after spending money.

Calibration: `estimate_tokens` accepts a `ratio` multiplier that can be derived
from real provider usage via `calibrate()`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

try:  # optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    tiktoken = None  # type: ignore


_ASCII = re.compile(r"[A-Za-z0-9]")
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_CJK = re.compile(r"[\u3000-\u9fff\uac00-\ud7af]")
_WHITESPACE = re.compile(r"\s")
_CODEISH = re.compile(r"[{}()\[\];:=<>|&$#@\\]")

_MODEL_CHARS_PER_TOKEN = {
    # name prefix -> chars per token for mixed English/code text
    "gpt-4": 3.9,
    "gpt-3.5": 3.9,
    "o1": 3.9,
    "o3": 3.9,
    "claude": 3.8,
    "llama": 3.7,
    "qwen": 3.5,
    "mistral": 3.7,
}


@lru_cache(maxsize=4096)
def _tiktoken_encoding(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def estimate_tokens(text: str, model: str = "", ratio: float = 1.0) -> int:
    """Return a token estimate for `text`.

    `ratio` scales the result (use values measured with `calibrate`).
    """
    if not text:
        return 0

    encoding = _tiktoken_encoding(model) if model else None
    if encoding is not None:
        return max(1, int(len(encoding.encode(text)) * ratio))

    ascii_chars = len(_ASCII.findall(text))
    cyrillic = len(_CYRILLIC.findall(text))
    cjk = len(_CJK.findall(text))
    codeish = len(_CODEISH.findall(text))
    whitespace = len(_WHITESPACE.findall(text))
    total = len(text)

    other = max(0, total - ascii_chars - cyrillic - cjk - whitespace)

    base = _MODEL_CHARS_PER_TOKEN.get(_model_family(model), 4.0)
    # Code punctuation and non-Latin scripts tokenize denser than prose.
    cost = (
        ascii_chars / base
        + cyrillic / 2.2
        + cjk / 1.5
        + other / 3.0
        + codeish / 2.0
        + whitespace / 6.0
    )
    return max(1, int(round(cost * ratio)))


def _model_family(model: str) -> str:
    lowered = model.lower()
    for prefix in _MODEL_CHARS_PER_TOKEN:
        if lowered.startswith(prefix):
            return prefix
    return ""


def messages_tokens(
    messages: Iterable, model: str = "", ratio: float = 1.0
) -> int:
    """Estimate the prompt tokens for a list of messages.

    Mirrors the chat format overhead: 4 tokens per message + 3 for the reply
    primer. Accepts Message objects or dicts.
    """
    total = 3
    for message in messages:
        if hasattr(message, "content"):
            content = message.content  # type: ignore[attr-defined]
            role = message.role  # type: ignore[attr-defined]
        else:
            content = message.get("content", "")
            role = message.get("role", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
        total += 4 + estimate_tokens(str(content), model, ratio)
        total += estimate_tokens(str(role), model, ratio)
    return total


def calibrate(estimated: int, actual: int) -> float:
    """Return a ratio to pass into `estimate_tokens` to match provider counts."""
    if estimated <= 0 or actual <= 0:
        return 1.0
    return round(actual / estimated, 4)


def cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    cost_in_per_million: float,
    cost_out_per_million: float,
) -> float:
    return round(
        prompt_tokens / 1_000_000 * cost_in_per_million
        + completion_tokens / 1_000_000 * cost_out_per_million,
        6,
    )
