"""Configuration for SOLAR.

Values come from (in order): explicit arguments, a JSON config file, environment
variables, built-in defaults. Everything is plain data so it can be inspected
and dumped back to disk with `solar config show`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "name": "gpt-4o-mini",
        "provider": "openai",
        "cost_in": 0.15,
        "cost_out": 0.60,
        "context_window": 128_000,
        "quality": 3,
        "tags": ["cheap", "general"],
    },
    {
        "name": "gpt-4o",
        "provider": "openai",
        "cost_in": 2.50,
        "cost_out": 10.00,
        "context_window": 128_000,
        "quality": 5,
        "tags": ["strong", "general"],
    },
    {
        "name": "claude-3-5-sonnet",
        "provider": "anthropic",
        "cost_in": 3.00,
        "cost_out": 15.00,
        "context_window": 200_000,
        "quality": 5,
        "tags": ["strong", "code"],
    },
    {
        "name": "qwen2.5-coder:7b",
        "provider": "ollama",
        "cost_in": 0.0,
        "cost_out": 0.0,
        "context_window": 32_000,
        "quality": 3,
        "tags": ["local", "code", "free"],
    },
    {
        "name": "llama3.1:8b",
        "provider": "ollama",
        "cost_in": 0.0,
        "cost_out": 0.0,
        "context_window": 128_000,
        "quality": 2,
        "tags": ["local", "general", "free"],
    },
]


@dataclass
class BudgetConfig:
    """Spending limits. 0 means "unlimited"."""

    max_tokens_per_request: int = 0
    max_tokens_per_session: int = 0
    max_tokens_per_project: int = 0
    max_tokens_per_day: int = 0
    warn_at_pct: float = 80.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CacheConfig:
    enabled: bool = True
    similarity_threshold: float = 0.92
    ttl_seconds: int = 7 * 24 * 3600
    max_entries: int = 5000
    # "safe" only reuses entries explicitly marked verified.
    mode: str = "safe"


@dataclass
class CompressionConfig:
    level: str = "balanced"  # conservative | balanced | aggressive
    min_savings_pct: float = 5.0
    min_keep_ratio: float = 0.30
    compress_assistant_history: bool = True
    keep_last_user_message_intact: bool = False
    compress_logs: bool = True
    max_log_lines: int = 40


@dataclass
class ContextConfig:
    enabled: bool = True
    max_files: int = 6
    max_snippet_lines: int = 60
    budget_tokens: int = 1500
    include_git_diff: bool = True
    max_diff_chars: int = 20_000
    include_dependencies: bool = True
    skip_dirs: list[str] = field(
        default_factory=lambda: [
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "target",
            "dist",
            "build",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        ]
    )


@dataclass
class RouterConfig:
    enabled: bool = False
    default_model: str = "gpt-4o-mini"
    # quality tier required per detected task kind
    quality_by_task: dict[str, int] = field(
        default_factory=lambda: {
            "simple": 1,
            "docs": 2,
            "code": 3,
            "debug": 4,
            "architecture": 5,
        }
    )
    rules: list[dict[str, str]] = field(default_factory=list)


@dataclass
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8420
    upstream_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    request_timeout: int = 300
    add_report_header: bool = True
    api_key: str = ""


@dataclass
class SolarConfig:
    db_path: str = "solar.db"
    project: str = "default"
    model: str = "gpt-4o-mini"
    prompt_version: str = "v1"

    budget: BudgetConfig = field(default_factory=BudgetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    models: list[dict[str, Any]] = field(default_factory=lambda: [m.copy() for m in DEFAULT_MODELS])

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "SolarConfig":
        data: dict[str, Any] = {}
        if path:
            raw = Path(path).read_text(encoding="utf-8")
            data = json.loads(raw)
        elif Path("solar.json").exists():
            data = json.loads(Path("solar.json").read_text(encoding="utf-8"))

        config = cls()
        for key, value in data.items():
            if not hasattr(config, key):
                continue
            current = getattr(config, key)
            if key in {"budget", "cache", "compression", "context", "router", "gateway"}:
                setattr(config, key, type(current)(**value))
            else:
                setattr(config, key, value)

        config._apply_env()
        return config

    def _apply_env(self) -> None:
        env = os.environ
        self.db_path = env.get("SOLAR_DB", self.db_path)
        self.project = env.get("SOLAR_PROJECT", self.project)
        self.model = env.get("SOLAR_MODEL", self.model)
        if env.get("SOLAR_PORT"):
            self.gateway.port = int(env["SOLAR_PORT"])
        if env.get("SOLAR_HOST"):
            self.gateway.host = env["SOLAR_HOST"]
        if env.get("OPENAI_BASE_URL"):
            self.gateway.upstream_base_url = env["OPENAI_BASE_URL"]
        if env.get("OLLAMA_BASE_URL"):
            self.gateway.ollama_base_url = env["OLLAMA_BASE_URL"]
        if env.get("SOLAR_API_KEY"):
            self.gateway.api_key = env["SOLAR_API_KEY"]
        if env.get("SOLAR_CACHE_MODE"):
            self.cache.mode = env["SOLAR_CACHE_MODE"]
        if env.get("SOLAR_COMPRESSION_LEVEL"):
            self.compression.level = env["SOLAR_COMPRESSION_LEVEL"]
        if env.get("SOLAR_ROUTER") in {"1", "true", "yes"}:
            self.router.enabled = True

    def dump(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(
            json.dumps(self.dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
