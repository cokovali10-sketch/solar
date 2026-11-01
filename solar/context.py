"""Project context extraction for SOLAR.

The goal is not "send the whole repo". The goal is: send the smallest set of
snippets that lets the model answer correctly. Everything is ranked by
relevance to the user query and trimmed to a token budget.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import ContextConfig
from .tokens import estimate_tokens

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala",
    ".sh", ".bash", ".zsh", ".ps1", ".sql", ".yaml", ".yml", ".toml",
    ".json", ".md", ".txt", ".rst", ".html", ".css", ".scss",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "as", "at", "by", "from", "into", "about",
    "i", "you", "we", "they", "he", "she", "my", "your", "our", "their",
    "me", "us", "them", "do", "does", "did", "can", "could", "should", "would",
    "will", "may", "might", "must", "not", "no", "so", "if", "then", "else",
    "when", "where", "why", "how", "what", "which", "who", "whom", "whose",
    "есть", "был", "была", "были", "это", "что", "как", "где", "когда",
    "для", "или", "и", "но", "не", "на", "в", "с", "по", "к", "из", "от",
    "до", "за", "о", "об", "про", "при", "без", "над", "под", "у", "же",
}


@dataclass
class FileCandidate:
    path: Path
    rel: str
    score: float = 0.0
    changed: bool = False
    snippet: str = ""
    tokens: int = 0
    reason: str = ""


@dataclass
class ContextBundle:
    root: str
    query: str
    files: list[dict] = field(default_factory=list)
    diff: str = ""
    structure: str = ""
    text: str = ""
    tokens: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "root": self.root,
            "query": self.query,
            "files": self.files,
            "diff": self.diff,
            "structure": self.structure,
            "text": self.text,
            "tokens": self.tokens,
            "notes": self.notes,
        }


def _run_git(root: Path, args: list[str], timeout: int = 20) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root)] + args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode != 0:
            return ""
        return result.stdout
    except Exception:
        return ""


def is_git_repo(root: Path) -> bool:
    if not (root / ".git").exists():
        return False
    toplevel = _run_git(root, ["rev-parse", "--show-toplevel"]).strip()
    try:
        return Path(toplevel).resolve() == root.resolve()
    except Exception:
        return False


def _query_terms(query: str) -> set[str]:
    terms = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u0430-\u044f\u0410-\u042f\u0401\u0451]{3,}", query))
    lowered = {term.lower() for term in terms}
    return {term for term in lowered if term not in _STOPWORDS}


def _read_text(path: Path, max_bytes: int = 256_000) -> str:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _structure(root: Path, max_entries: int = 80) -> str:
    lines: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            rel = Path(dirpath).relative_to(root)
            depth = len(rel.parts)
            if depth > 2:
                dirnames[:] = []
                continue
            # prune ignored dirs
            dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "target", "dist", "build", "__pycache__", ".venv", "venv"}]
            if depth == 0:
                prefix = ""
            else:
                prefix = "  " * (depth - 1) + "- "
            if depth > 0:
                lines.append(f"{prefix}{rel.name}/")
            for filename in sorted(filenames)[:20]:
                if depth < 2:
                    lines.append(f"{'  ' * depth}- {filename}")
            if len(lines) > max_entries:
                break
    except Exception:
        return ""
    return "\n".join(lines[:max_entries])


def _changed_files(root: Path) -> set[str]:
    changed: set[str] = set()
    status = _run_git(root, ["status", "--porcelain"])
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[-1]
        changed.add(path)
    diff_names = _run_git(root, ["diff", "--name-only", "HEAD"])
    for line in diff_names.splitlines():
        if line.strip():
            changed.add(line.strip())
    return changed


def _recent_files(root: Path, limit: int = 40) -> list[Path]:
    collected: list[tuple[float, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "target", "dist", "build", "__pycache__", ".venv", "venv", ".pytest_cache"}]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in _CODE_EXTENSIONS:
                continue
            try:
                collected.append((path.stat().st_mtime, path))
            except Exception:
                continue
    collected.sort(reverse=True)
    return [path for _, path in collected[:limit]]


def _candidate_files(root: Path, config: ContextConfig) -> list[Path]:
    if is_git_repo(root):
        tracked = _run_git(root, ["ls-files"])
        candidates = [root / line.strip() for line in tracked.splitlines() if line.strip()]
        candidates = [p for p in candidates if p.is_file() and p.suffix.lower() in _CODE_EXTENSIONS]
    else:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in set(config.skip_dirs)]
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.suffix.lower() in _CODE_EXTENSIONS:
                    candidates.append(path)
    # filter skip dirs even for git tracked
    filtered: list[Path] = []
    for path in candidates:
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in set(config.skip_dirs) for part in rel_parts):
            continue
        filtered.append(path)
    return filtered


def _score_file(path: Path, root: Path, content: str, terms: set[str], changed: bool, query: str) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []
    rel = str(path.relative_to(root)).lower()
    lowered = content.lower()
    rel_terms = set(re.findall(r"[a-z_][a-z0-9_]{2,}", rel))
    overlap = terms & rel_terms
    if overlap:
        score += 3.0 * len(overlap)
        reasons.append(f"path:{','.join(sorted(overlap)[:3])}")
    content_hits = sum(lowered.count(term) for term in terms)
    score += min(content_hits, 20) * 0.5
    if content_hits:
        reasons.append(f"content:{content_hits}")
    if changed:
        score += 6.0
        reasons.append("changed")
    if "test" in query.lower() and ("test" in rel or "spec" in rel):
        score += 4.0
        reasons.append("test")
    name = path.name.lower()
    if name in {"readme.md", "pyproject.toml", "package.json", "go.mod", "cargo.toml", "requirements.txt"}:
        score += 1.0
        reasons.append("manifest")
    return score, ";".join(reasons)


def _extract_snippet(content: str, terms: set[str], max_lines: int) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    if not terms:
        return "\n".join(lines[:max_lines])
    # find best starting line by term density in a sliding window
    best_start = 0
    best_score = -1
    window = min(max_lines, len(lines))
    for start in range(0, max(1, len(lines) - window + 1)):
        chunk = "\n".join(lines[start : start + window]).lower()
        score = sum(chunk.count(term) for term in terms)
        if score > best_score:
            best_score = score
            best_start = start
    if best_score <= 0:
        best_start = 0
    selected = lines[best_start : best_start + window]
    if best_start > 0:
        selected = [f"... (line {best_start + 1})"] + selected
    if best_start + window < len(lines):
        selected = selected + [f"... ({len(lines) - best_start - window} more lines)"]
    return "\n".join(selected)


_PY_IMPORT = re.compile(r"(?m)^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))")
_JS_IMPORT = re.compile(r"(?:from\s+['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]\s*\))")
_GO_IMPORT = re.compile(r'"([^"]+)"')


def _dependencies(path: Path, content: str) -> list[str]:
    deps: list[str] = []
    suffix = path.suffix.lower()
    if suffix == ".py":
        for match in _PY_IMPORT.finditer(content):
            name = match.group(1) or match.group(2)
            if name:
                deps.append(name.split(".")[0])
    elif suffix in {".js", ".ts", ".tsx", ".jsx"}:
        for match in _JS_IMPORT.finditer(content):
            name = match.group(1) or match.group(2)
            if name and name.startswith("."):
                deps.append(name)
    elif suffix == ".go":
        for match in _GO_IMPORT.finditer(content):
            deps.append(match.group(1))
    # de-dupe preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for dep in deps:
        if dep not in seen:
            seen.add(dep)
            unique.append(dep)
    return unique


def _resolve_dependency(root: Path, owner: Path, dep: str) -> Path | None:
    if dep.startswith("."):
        candidate = (owner.parent / dep).resolve()
        for suffix in ("", ".py", ".js", ".ts", ".tsx", ".go", "/index.js", "/index.ts"):
            path = Path(str(candidate) + suffix)
            if path.is_file():
                return path
        return None
    # python module -> file
    module_path = root / (dep.replace(".", "/") + ".py")
    if module_path.is_file():
        return module_path
    package_init = root / dep.replace(".", "/") / "__init__.py"
    if package_init.is_file():
        return package_init
    return None


def _public_summary(path: Path, max_lines: int = 6) -> str:
    content = _read_text(path)
    if not content:
        return ""
    defs = re.findall(r"(?m)^(?:def |class |func |fn |function |export (?:function|class) ).*$", content)
    if not defs:
        return ""
    return "\n".join(defs[:max_lines])


def build_context(
    root: str | os.PathLike[str],
    query: str,
    config: ContextConfig | None = None,
    model: str = "",
) -> ContextBundle:
    config = config or ContextConfig()
    root_path = Path(root).resolve()
    bundle = ContextBundle(root=str(root_path), query=query)
    if not root_path.exists():
        bundle.notes.append("project root does not exist")
        return bundle

    bundle.structure = _structure(root_path)
    terms = _query_terms(query)
    changed = _changed_files(root_path) if is_git_repo(root_path) else set()

    candidates = _candidate_files(root_path, config)
    scored: list[FileCandidate] = []
    for path in candidates:
        content = _read_text(path)
        if not content:
            continue
        rel = str(path.relative_to(root_path))
        is_changed = rel in changed
        score, reason = _score_file(path, root_path, content, terms, is_changed, query)
        if score <= 0 and not is_changed:
            continue
        scored.append(FileCandidate(path=path, rel=rel, score=score, changed=is_changed, reason=reason))

    scored.sort(key=lambda item: (-item.score, item.rel))

    if not is_git_repo(root_path):
        bundle.notes.append("not a git repository; using mtime-based recent files")
        recent = _recent_files(root_path, limit=config.max_files * 2)
        for path in recent:
            rel = str(path.relative_to(root_path))
            if rel in {c.rel for c in scored}:
                continue
            scored.append(FileCandidate(path=path, rel=rel, score=0.5, changed=False, reason="recent"))

    # Build snippets within budget
    selected: list[FileCandidate] = []
    used_tokens = 0
    budget = max(500, config.budget_tokens)
    for candidate in scored[: config.max_files * 4]:
        if len(selected) >= config.max_files:
            break
        content = _read_text(candidate.path)
        if not content:
            continue
        snippet = _extract_snippet(content, terms, config.max_snippet_lines)
        # dependency summaries
        dep_summaries: list[str] = []
        if config.include_dependencies:
            for dep in _dependencies(candidate.path, content):
                resolved = _resolve_dependency(root_path, candidate.path, dep)
                if resolved is None or resolved == candidate.path:
                    continue
                summary = _public_summary(resolved)
                if summary:
                    dep_summaries.append(f"# {resolved.relative_to(root_path)}\n{summary}")
        full_snippet = snippet
        if dep_summaries:
            full_snippet = snippet + "\n\n# dependencies\n" + "\n".join(dep_summaries)
        candidate.snippet = full_snippet
        candidate.tokens = estimate_tokens(full_snippet, model) + 20
        if used_tokens + candidate.tokens > budget and selected:
            bundle.notes.append(f"budget reached after {len(selected)} files")
            break
        selected.append(candidate)
        used_tokens += candidate.tokens

    bundle.files = [
        {
            "path": item.rel,
            "score": round(item.score, 2),
            "changed": item.changed,
            "reason": item.reason,
            "tokens": item.tokens,
            "snippet": item.snippet,
        }
        for item in selected
    ]

    # diff
    if config.include_git_diff and is_git_repo(root_path):
        diff = _run_git(root_path, ["diff", "HEAD"])
        if not diff.strip():
            diff = _run_git(root_path, ["diff"])
        if len(diff) > config.max_diff_chars:
            diff = diff[: config.max_diff_chars] + "\n... [truncated]"
        bundle.diff = diff

    # assemble text
    sections: list[str] = []
    if bundle.structure:
        sections.append("## Project structure\n" + bundle.structure)
    if bundle.files:
        file_section = "## Relevant files\n"
        for item in bundle.files:
            file_section += f"\n### {item['path']} (score {item['score']})\n```\n{item['snippet']}\n```\n"
        sections.append(file_section)
    if bundle.diff.strip():
        sections.append("## Git diff\n```diff\n" + bundle.diff.strip() + "\n```")

    bundle.text = "\n\n".join(sections).strip()
    bundle.tokens = estimate_tokens(bundle.text, model)
    if not bundle.text:
        bundle.notes.append("no context collected")
    return bundle


def inject_context(messages: list, bundle: ContextBundle) -> list:
    """Return a new message list with the context injected.

    Preference order: append to an existing system message; otherwise insert a
    new system message at the front. The user's last message is never modified.
    """
    if not bundle.text:
        return [m.copy() if hasattr(m, "copy") else m for m in messages]
    from .models import Message

    new_messages = [Message(m.role, m.content) if not isinstance(m, Message) else Message(m.role, m.content) for m in messages]
    context_block = "Project context (auto-generated by SOLAR):\n" + bundle.text
    system_indices = [i for i, m in enumerate(new_messages) if m.role == "system"]
    if system_indices:
        idx = system_indices[0]
        new_messages[idx] = Message("system", new_messages[idx].content + "\n\n" + context_block)
    else:
        new_messages.insert(0, Message("system", context_block))
    return new_messages
