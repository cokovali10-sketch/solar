"""Protection layer: split text into compressible and non-compressible spans.

The rule is simple and non-negotiable: code, commands, identifiers, paths,
numbers and quoted literals are never rewritten. Only prose between those spans
is eligible for compression. This is what keeps "token savings" from turning
into "the model invented an API that does not exist".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

FENCED_CODE = r"```[\s\S]*?```"
INLINE_CODE = r"`[^`\n]{1,400}`"
URL = r"https?://[^\s<>\"')]+"
WINDOWS_PATH = r"\b[A-Za-z]:\\[^\s\"'<>|]+"
UNIX_PATH = r"(?<![\w.])(?:\.{1,2}/|/)(?:[\w.@+\-]+/)*[\w.@+\-]+\.[A-Za-z0-9]{1,10}\b"
QUOTED = r"\"[^\"\n]{1,300}\"|'[^'\n]{1,300}'"
FUNCTION_CALL = r"\b[A-Za-z_][\w.]{1,60}\("
IDENTIFIER = (
    r"\b(?:[a-z][a-z0-9]*_[a-zA-Z0-9_]+"  # snake_case
    r"|[a-z]+[A-Z][a-zA-Z0-9]*"  # camelCase
    r"|[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*)\b"  # PascalCase with 2+ humps
)
ENV_VAR = r"\b[A-Z][A-Z0-9_]{2,}\b"
NUMBER = r"(?<![\w.])\d+(?:\.\d+)?(?:ms|s|m|h|KB|MB|GB|TB|%|px)?\b"
SHELL_LINE = (
    r"^[ \t]*(?:\$|>|PS>)[ \t].+$"
    r"|^[ \t]*(?:git|npm|pnpm|yarn|go|cargo|docker|kubectl|helm|python|pip|pytest"
    r"|make|dotnet|java|mvn|gradle|rustc|node|npx)[ \t][^\n]*$"
)

# NOTE: multiline mode is enabled once for the whole alternation. Python 3.11+
# rejects global inline flags in the middle of a pattern.
_PATTERN = re.compile(
    "(?m)"
    + "|".join(
        f"(?P<{name}>{pattern})"
        for name, pattern in [
            ("code", FENCED_CODE),
            ("inline_code", INLINE_CODE),
            ("url", URL),
            ("win_path", WINDOWS_PATH),
            ("unix_path", UNIX_PATH),
            ("quoted", QUOTED),
            ("shell", SHELL_LINE),
            ("call", FUNCTION_CALL),
            ("identifier", IDENTIFIER),
            ("env", ENV_VAR),
            ("number", NUMBER),
        ]
    )
)


@dataclass
class Part:
    text: str
    protected: bool = False
    kind: str = "prose"

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class ProtectedText:
    parts: list[Part] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts)

    @property
    def protected_parts(self) -> list[Part]:
        return [part for part in self.parts if part.protected]

    @property
    def protected_count(self) -> int:
        return len(self.protected_parts)

    def kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for part in self.protected_parts:
            counts[part.kind] = counts.get(part.kind, 0) + 1
        return counts

    def protected_texts(self) -> list[str]:
        return [part.text for part in self.protected_parts]


def split(text: str) -> ProtectedText:
    """Split `text` into protected and compressible parts."""
    if not text:
        return ProtectedText([Part("")])

    parts: list[Part] = []
    cursor = 0
    for match in _PATTERN.finditer(text):
        if match.start() > cursor:
            parts.append(Part(text[cursor : match.start()]))
        parts.append(Part(match.group(0), protected=True, kind=match.lastgroup or "unknown"))
        cursor = match.end()
    if cursor < len(text):
        parts.append(Part(text[cursor:]))

    return ProtectedText(_merge(parts))


def _merge(parts: Iterable[Part]) -> list[Part]:
    merged: list[Part] = []
    for part in parts:
        if not part.text:
            continue
        if merged and merged[-1].protected == part.protected and merged[-1].kind == part.kind:
            merged[-1] = Part(
                merged[-1].text + part.text, part.protected, part.kind
            )
        else:
            merged.append(Part(part.text, part.protected, part.kind))
    return merged or [Part("")]


def integrity_check(protected: ProtectedText, compressed_text: str) -> tuple[bool, list[str]]:
    """Verify every protected span survived compression verbatim."""
    missing: list[str] = []
    for span in protected.protected_texts():
        if span and span not in compressed_text:
            missing.append(span)
    return (not missing, missing)


def restore(parts: Iterable[Part]) -> str:
    return "".join(part.text for part in parts)
