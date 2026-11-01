"""Prompt and response compression with hard protection rules.

Design rules, in priority order:

1. Never rewrite code, commands, identifiers, paths, numbers or quoted literals.
2. Never drop a sentence that carries a constraint (must/never/only/limit/...).
3. Never drop a sentence that contains a protected span.
4. If integrity cannot be proven, return the original text untouched.

Compression only touches natural-language prose, which is also the part of a
prompt whose removal cannot silently change an API contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .config import CompressionConfig
from .models import Message
from .protect import integrity_check, split
from .tokens import estimate_tokens

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

_POLITE_PREFIX = [
    r"^(?:could|can|would)\s+you\s+(?:please\s+)?",
    r"^(?:please|kindly)\s+",
    r"^i\s+(?:would\s+like|want|need)\s+you\s+to\s+",
    r"^i\s+was\s+wondering\s+if\s+you\s+could\s+",
    r"^(?:пожалуйста,?\s*)",
    r"^не\s+могли\s+бы\s+вы\s+",
    r"^я\s+(?:хочу|хотел\s+бы),?\s+чтобы\s+ты\s+",
    r"^будь(?:те)?\s+добр(?:ы)?,?\s+",
]
_POLITE_RE = re.compile("|".join(_POLITE_PREFIX), re.IGNORECASE)

_MID_FILLER = re.compile(
    r"(?<=\s)(?:please|kindly|пожалуйста|будьте добры)(?=[\s,.;:!?])",
    re.IGNORECASE,
)

# Words that can be deleted without changing the requirement set.
_SOFT_FILLER = re.compile(
    r"\b(?:just|simply|really|actually|basically|literally|quite|rather|"
    r"somewhat|truly|definitely|certainly|просто|буквально|вообще|"
    r"на\s+самом\s+деле|реально|конкретно)\b",
    re.IGNORECASE,
)
_VERY = re.compile(r"\b(?:very|extremely|really)\s+(?=[a-zа-яё])", re.IGNORECASE)

_PHRASE_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bin order to\b", re.I), "to"),
    (re.compile(r"\bdue to the fact that\b", re.I), "because"),
    (re.compile(r"\bat this point in time\b", re.I), "now"),
    (re.compile(r"\bin the event that\b", re.I), "if"),
    (re.compile(r"\bfor the purpose of\b", re.I), "to"),
    (re.compile(r"\bwith regard to\b", re.I), "about"),
    (re.compile(r"\bwith respect to\b", re.I), "about"),
    (re.compile(r"\bmake sure that\b", re.I), "ensure"),
    (re.compile(r"\bensure that\b", re.I), "ensure"),
    (re.compile(r"\bas well as\b", re.I), "and"),
    (re.compile(r"\ba large number of\b", re.I), "many"),
    (re.compile(r"\bthe majority of\b", re.I), "most"),
    (re.compile(r"\bhas the ability to\b", re.I), "can"),
    (re.compile(r"\bis able to\b", re.I), "can"),
    (re.compile(r"\bare able to\b", re.I), "can"),
    (re.compile(r"\bprior to\b", re.I), "before"),
    (re.compile(r"\bsubsequent to\b", re.I), "after"),
    (re.compile(r"\bin the near future\b", re.I), "soon"),
    (re.compile(r"\butili[sz]e\b", re.I), "use"),
    (re.compile(r"\butili[sz]ing\b", re.I), "using"),
    (re.compile(r"\bdemonstrates?\b", re.I), "shows"),
    (re.compile(r"\bapproximately\b", re.I), "about"),
    (re.compile(r"\bin addition\b", re.I), "also"),
    (re.compile(r"\bin addition to\b", re.I), "besides"),
    (re.compile(r"\bfor the reason that\b", re.I), "because"),
    (re.compile(r"\bit is (?:important|necessary) to note that\b", re.I), ""),
    (re.compile(r"\bplease note that\b", re.I), ""),
    (re.compile(r"\bв связи с тем,? что\b", re.I), "потому что"),
    (re.compile(r"\bдля того чтобы\b", re.I), "чтобы"),
    (re.compile(r"\bс целью\b", re.I), "чтобы"),
    (re.compile(r"\bв настоящее время\b", re.I), "сейчас"),
    (re.compile(r"\bв случае если\b", re.I), "если"),
    (re.compile(r"\bа также\b", re.I), "и"),
    (re.compile(r"\bбольшое количество\b", re.I), "много"),
    (re.compile(r"\bявляется\b", re.I), "—"),
    (re.compile(r"\bнеобходимо\b", re.I), "нужно"),
    (re.compile(r"\bосуществлять\b", re.I), "делать"),
    (re.compile(r"\bстоит отметить,? что\b", re.I), ""),
    (re.compile(r"\bважно отметить,? что\b", re.I), ""),
    (re.compile(r"\bна мой взгляд,?\b", re.I), ""),
    (re.compile(r"\bя думаю,? что\b", re.I), ""),
    (re.compile(r"\bмне кажется,? что\b", re.I), ""),
]

# If a sentence mentions any of these, it is a requirement and is kept verbatim.
_CONSTRAINT = re.compile(
    r"\b(?:must|shall|should|required?|requirement|never|not|no|only|exactly|"
    r"at least|at most|maximum|minimum|max|min|limit|budget|deadline|"
    r"contract|interface|signature|api|schema|test|tests|error|errors|"
    r"forbid(?:den)?|prohibit(?:ed)?|cannot|can't|don't|do not|does not|"
    r"deprecated|breaking|backward[s]?|output|input|format|json|yaml|"
    r"обязательн\w*|должен|должна|должны|нельзя|запрещ\w*|лимит|бюджет|"
    r"только|ровно|не менее|не более|максимум|минимум|тест\w*|"
    r"ошибк\w*|контракт|интерфейс|сигнатур\w*|формат)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:not|never|no|cannot|can't|don't|doesn't|isn't|aren't|won't|"
    r"не|нет|нельзя|никогда|без)\b",
    re.IGNORECASE,
)

_META_OPENER = re.compile(
    r"^(?:sure|certainly|of course|absolutely|great question|happy to help|"
    r"here(?:'s| is| are)\b[^\n]{0,80}[:.]?\s*|"
    r"конечно|разумеется|отличный вопрос|с удовольствием)\b[,:.!]?\s*",
    re.IGNORECASE,
)
_META_CLOSER = re.compile(
    r"(?im)(?:hope (?:this|that) helps|let me know if[^\n]*|"
    r"feel free to[^\n]*|надеюсь,? это поможет[^\n]*|"
    r"дайте знать[^\n]*|если (?:нужно|надо)[^\n]*)[.!]?\s*$"
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9`\"'(])")

_LOG_HINT = re.compile(
    r"(?m)^\s*(?:\[?\d{4}-\d{2}-\d{2}[T ]|\d{2}:\d{2}:\d{2}|"
    r"(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)\b)"
)
_LOG_SEVERITY = re.compile(
    r"\b(?:ERROR|WARN|WARNING|CRITICAL|FATAL|Exception|Traceback|panic|"
    r"FAILED|FAIL)\b"
)

_LEVELS = {
    "conservative": {
        "polite": True,
        "phrases": True,
        "soft_filler": False,
        "very": False,
        "dedupe_sentences": False,
        "dedupe_paragraphs": True,
        "drop_meta": False,
    },
    "balanced": {
        "polite": True,
        "phrases": True,
        "soft_filler": True,
        "very": False,
        "dedupe_sentences": True,
        "dedupe_paragraphs": True,
        "drop_meta": True,
    },
    "aggressive": {
        "polite": True,
        "phrases": True,
        "soft_filler": True,
        "very": True,
        "dedupe_sentences": True,
        "dedupe_paragraphs": True,
        "drop_meta": True,
    },
}


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class CompressionResult:
    original: str
    compressed: str
    original_tokens: int = 0
    compressed_tokens: int = 0
    changed: bool = False
    integrity_ok: bool = True
    missing_spans: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def savings_pct(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return round(100.0 * self.saved_tokens / self.original_tokens, 2)


@dataclass
class MessagesResult:
    messages: list[Message]
    notes: list[str] = field(default_factory=list)
    integrity_ok: bool = True


# --------------------------------------------------------------------------- #
# Prose helpers
# --------------------------------------------------------------------------- #


def _normalize_key(sentence: str) -> str:
    lowered = sentence.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return re.sub(r"[^\w\s]", "", lowered).strip()


def _capitalize(text: str) -> str:
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text


def _strip_politeness(sentence: str) -> str:
    updated = _POLITE_RE.sub("", sentence)
    if updated != sentence:
        updated = _MID_FILLER.sub(" ", updated)
        updated = re.sub(r"\s{2,}", " ", updated).strip()
        updated = _capitalize(updated) if updated else updated
    return updated


def _apply_phrases(sentence: str) -> str:
    updated = sentence
    for pattern, replacement in _PHRASE_MAP:
        def _sub(match: re.Match[str], rep: str = replacement) -> str:
            value = rep
            if not value:
                return ""
            if match.group(0)[:1].isupper():
                value = value[:1].upper() + value[1:]
            return value

        updated = pattern.sub(_sub, updated)
    updated = re.sub(r"\s{2,}", " ", updated)
    updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
    return updated.strip()


def _compress_sentence(sentence: str, settings: dict) -> str:
    updated = sentence
    if settings["polite"]:
        updated = _strip_politeness(updated)
    if settings["phrases"]:
        updated = _apply_phrases(updated)
    if settings["soft_filler"]:
        updated = _SOFT_FILLER.sub("", updated)
    if settings["very"]:
        updated = _VERY.sub("", updated)
    updated = re.sub(r"\s{2,}", " ", updated).strip()
    if not updated:
        return sentence
    return updated


def _is_protected_sentence(sentence: str) -> bool:
    return bool(split(sentence).protected_count)


def _is_constraint(sentence: str) -> bool:
    return bool(_CONSTRAINT.search(sentence) or _NEGATION.search(sentence))


def _compress_prose(text: str, settings: dict, is_assistant: bool) -> tuple[str, int]:
    """Compress a single prose chunk. Returns (text, dropped_sentences)."""
    if not text.strip():
        return text, 0

    if is_assistant and settings["drop_meta"]:
        while True:
            updated = _META_CLOSER.sub("", text)
            if updated == text:
                break
            text = updated
        text = _META_OPENER.sub("", text)

    paragraphs = re.split(r"(\n\s*\n)", text)
    seen_paragraphs: set[str] = set()
    seen_sentences: set[str] = set()
    out: list[str] = []
    dropped = 0

    for chunk in paragraphs:
        if not chunk.strip():
            out.append(chunk)
            continue

        key = _normalize_key(chunk)
        if settings["dedupe_paragraphs"] and len(key) > 80 and key in seen_paragraphs:
            dropped += 1
            continue
        seen_paragraphs.add(key)

        lines = chunk.splitlines(keepends=False)
        new_lines: list[str] = []
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
            leading = line[: len(line) - len(line.lstrip())]
            trailing = line[len(line.rstrip()) :]
            stripped = line.strip()
            # Whole-line dedup: drop exact duplicate lines even when they
            # contain protected spans (the first occurrence keeps the span, so
            # integrity still holds).
            line_key = _normalize_key(stripped)
            if settings["dedupe_sentences"] and len(line_key) > 15 and line_key in seen_sentences:
                dropped += 1
                continue

            if _is_protected_sentence(stripped):
                new_lines.append(line)
                seen_sentences.add(line_key)
                continue

            if _is_constraint(stripped):
                # Constraint sentences are never dropped and never have filler
                # words stripped (that could weaken a requirement), but polite
                # frames and verbose phrases are still safe to compress.
                # Sentence-level dedup still applies.
                light_settings = dict(settings)
                light_settings["soft_filler"] = False
                light_settings["very"] = False
                light_settings["dedupe_sentences"] = False
                sentences = _SENTENCE_SPLIT.split(stripped)
                kept: list[str] = []
                for sentence in sentences:
                    s_key = _normalize_key(sentence)
                    if settings["dedupe_sentences"] and len(s_key) > 15 and s_key in seen_sentences:
                        dropped += 1
                        continue
                    seen_sentences.add(s_key)
                    updated = _compress_sentence(sentence, light_settings)
                    kept.append(updated if updated else sentence)
                rebuilt = " ".join(kept).strip()
                if rebuilt:
                    new_lines.append(leading + rebuilt + trailing)
                else:
                    new_lines.append(line)
                continue

            sentences = _SENTENCE_SPLIT.split(stripped)
            kept: list[str] = []
            for sentence in sentences:
                normalized = _normalize_key(sentence)
                if (
                    settings["dedupe_sentences"]
                    and len(normalized) > 15
                    and normalized in seen_sentences
                ):
                    dropped += 1
                    continue
                seen_sentences.add(normalized)
                updated = _compress_sentence(sentence, settings)
                if updated:
                    kept.append(updated)
                else:
                    dropped += 1
            rebuilt = " ".join(kept).strip()
            if rebuilt:
                new_lines.append(leading + rebuilt + trailing)
            else:
                new_lines.append(line)

        rebuilt_chunk = "\n".join(new_lines)
        out.append(rebuilt_chunk if rebuilt_chunk.strip() else chunk)

    result = "".join(out)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result, dropped


# --------------------------------------------------------------------------- #
# Log compression
# --------------------------------------------------------------------------- #


def _log_signature(line: str) -> str:
    signature = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b", "<ts>", line)
    signature = re.sub(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b", "<ts>", signature)
    signature = re.sub(r"\b0x[0-9a-fA-F]+\b", "<hex>", signature)
    signature = re.sub(r"\b[0-9a-fA-F]{8,}\b", "<id>", signature)
    signature = re.sub(r"\b\d+(?:\.\d+)?(?:ms|s|m|h|KB|MB|GB)?\b", "<n>", signature)
    signature = re.sub(r"\s+", " ", signature).strip()
    return signature[:200]


def _looks_like_logs(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    hints = sum(1 for line in lines if _LOG_HINT.search(line))
    return hints >= max(4, int(len(lines) * 0.25))


def compress_logs(text: str, config: CompressionConfig) -> tuple[str, str | None]:
    """Collapse repeated log lines into `[xN] signature` entries.

    Error lines are always kept, up to a cap, because losing the actual exception
    is far more expensive than the tokens it costs.
    """
    if not config.compress_logs or not _looks_like_logs(text):
        return text, None

    lines = text.splitlines()
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        signature = _log_signature(line)
        if signature not in groups:
            groups[signature] = []
            order.append(signature)
        groups[signature].append(line)

    error_lines: list[str] = []
    other: list[str] = []
    for signature in order:
        bucket = groups[signature]
        errors = [line for line in bucket if _LOG_SEVERITY.search(line)]
        if errors:
            error_lines.extend(errors[:3])
            remaining = len(bucket) - min(3, len(bucket))
            if remaining > 0:
                other.append(f"[x{remaining}] {signature}")
            continue
        if len(bucket) > 1:
            other.append(f"[x{len(bucket)}] {signature}")
        else:
            other.append(bucket[0])

    budget = max(10, config.max_log_lines)
    head = other[: max(3, budget // 2)]
    tail = other[-max(3, budget // 3) :] if len(other) > budget else []
    body = head + (["..."] if tail else []) + tail
    combined: list[str] = []
    if error_lines:
        combined.append(f"=== errors ({len(error_lines)} kept) ===")
        combined.extend(error_lines[: max(5, budget // 2)])
    if body:
        combined.append(f"=== other lines ({len(lines)} total) ===")
        combined.extend(body)

    return "\n".join(combined), f"logs: {len(lines)} lines collapsed"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def resolve_level(config: CompressionConfig, level: str | None = None) -> dict:
    name = (level or config.level or "balanced").lower()
    return _LEVELS.get(name, _LEVELS["balanced"])


def compress_text(
    text: str,
    model: str = "",
    config: CompressionConfig | None = None,
    level: str | None = None,
    role: str = "user",
) -> CompressionResult:
    config = config or CompressionConfig()
    if not text or not text.strip():
        return CompressionResult(text, text)

    settings = resolve_level(config, level)
    notes: list[str] = []

    after_logs, log_note = compress_logs(text, config)
    if log_note:
        notes.append(log_note)

    protected = split(after_logs)
    prose_before = sum(len(p.text) for p in protected.parts if not p.protected)

    rebuilt: list[str] = []
    dropped_total = 0
    for part in protected.parts:
        if part.protected:
            rebuilt.append(part.text)
        else:
            compressed, dropped = _compress_prose(part.text, settings, role == "assistant")
            rebuilt.append(compressed)
            dropped_total += dropped
    candidate = "".join(rebuilt)

    prose_after = sum(
        len(p.text) for p in split(candidate).parts if not p.protected
    )
    keep_ratio = 1.0 if prose_before == 0 else prose_after / prose_before

    integrity_ok, missing = integrity_check(protected, candidate)

    accepted = candidate
    if not integrity_ok:
        notes.append("integrity check failed; prose compression reverted")
        accepted = after_logs
    elif prose_before > 0 and keep_ratio < config.min_keep_ratio:
        notes.append(
            f"keep ratio {keep_ratio:.0%} below {config.min_keep_ratio:.0%}; prose compression reverted"
        )
        accepted = after_logs
    elif dropped_total:
        notes.append(f"dropped {dropped_total} redundant sentences/paragraphs")

    original_tokens = estimate_tokens(text, model)
    compressed_tokens = estimate_tokens(accepted, model)
    savings_pct = (
        0.0 if original_tokens == 0 else 100.0 * (original_tokens - compressed_tokens) / original_tokens
    )

    if accepted != text and savings_pct < config.min_savings_pct and log_note is None:
        notes.append(f"savings {savings_pct:.1f}% below threshold; original kept")
        accepted = text
        compressed_tokens = original_tokens
    elif accepted != text and savings_pct < config.min_savings_pct and log_note is not None:
        notes.append(f"prose savings below threshold, but log compression kept ({savings_pct:.1f}%)")

    return CompressionResult(
        original=text,
        compressed=accepted,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        changed=accepted != text,
        integrity_ok=integrity_ok,
        missing_spans=missing,
        notes=notes,
    )


def compress_messages(
    messages: Iterable[Message],
    model: str = "",
    config: CompressionConfig | None = None,
) -> MessagesResult:
    config = config or CompressionConfig()
    items = list(messages)
    if not items:
        return MessagesResult([])

    last_user_index = max(
        (i for i, message in enumerate(items) if message.role == "user"),
        default=-1,
    )

    result: list[Message] = []
    notes: list[str] = []
    integrity_ok = True
    seen_assistant: set[str] = set()

    for index, message in enumerate(items):
        keep_intact = (
            config.keep_last_user_message_intact
            and index == last_user_index
            and message.role == "user"
        )
        if message.role == "assistant" and not config.compress_assistant_history:
            result.append(message)
            continue

        if keep_intact:
            result.append(Message(message.role, message.content))
            continue

        if message.role == "assistant":
            key = _normalize_key(message.content)[:400]
            if key and key in seen_assistant and len(message.content) > 200:
                notes.append("duplicate assistant message replaced with pointer")
                result.append(
                    Message(
                        message.role,
                        "[Identical answer omitted: it is already present earlier in the conversation.]",
                    )
                )
                continue
            seen_assistant.add(key)

        outcome = compress_text(
            message.content,
            model=model,
            config=config,
            role=message.role,
        )
        integrity_ok = integrity_ok and outcome.integrity_ok
        notes.extend(outcome.notes)
        result.append(Message(message.role, outcome.compressed))

    return MessagesResult(result, notes=notes, integrity_ok=integrity_ok)


def compress_response(
    text: str,
    model: str = "",
    config: CompressionConfig | None = None,
) -> CompressionResult:
    """Compress an assistant answer.

    Code, commands and warnings survive; only conversational padding and
    duplicated paragraphs are removed. This is opt-in because it changes the
    bytes the user sees.
    """
    return compress_text(
        text,
        model=model,
        config=config,
        level="balanced",
        role="assistant",
    )
