"""Token and cost budget management for SOLAR."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import BudgetConfig
from .storage import Storage


@dataclass
class BudgetDecision:
    allowed: bool
    reason: str = ""
    remaining_tokens: int = 0
    remaining_cost: float = 0.0
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class BudgetManager:
    def __init__(self, storage: Storage, config: BudgetConfig):
        self.storage = storage
        self.config = config

    def check(
        self,
        estimated_tokens: int,
        estimated_cost: float = 0.0,
        session_id: str | None = None,
    ) -> BudgetDecision:
        warnings: list[str] = []
        cfg = self.config

        # per request
        if cfg.max_tokens_per_request and estimated_tokens > cfg.max_tokens_per_request:
            return BudgetDecision(
                False,
                f"request needs {estimated_tokens} tokens, limit is {cfg.max_tokens_per_request}",
                0, 0.0, warnings,
            )

        # per session
        if cfg.max_tokens_per_session and session_id:
            session_tokens = self._session_tokens(session_id)
            if session_tokens + estimated_tokens > cfg.max_tokens_per_session:
                return BudgetDecision(
                    False,
                    f"session would exceed {cfg.max_tokens_per_session} tokens "
                    f"(used {session_tokens})",
                    max(0, cfg.max_tokens_per_session - session_tokens),
                    0.0, warnings,
                )
            remaining = cfg.max_tokens_per_session - (session_tokens + estimated_tokens)
            if cfg.warn_at_pct and cfg.max_tokens_per_session:
                used_pct = 100.0 * (session_tokens + estimated_tokens) / cfg.max_tokens_per_session
                if used_pct >= cfg.warn_at_pct:
                    warnings.append(f"session usage at {used_pct:.0f}% of limit")

        # per project (all time, since project start)
        if cfg.max_tokens_per_project:
            totals = self.storage.budget_totals(self.storage.project)
            if totals["tokens"] + estimated_tokens > cfg.max_tokens_per_project:
                return BudgetDecision(
                    False,
                    f"project would exceed {cfg.max_tokens_per_project} tokens "
                    f"(used {totals['tokens']})",
                    max(0, cfg.max_tokens_per_project - totals["tokens"]),
                    0.0, warnings,
                )
            remaining_project = cfg.max_tokens_per_project - (totals["tokens"] + estimated_tokens)
            if cfg.warn_at_pct:
                used_pct = 100.0 * (totals["tokens"] + estimated_tokens) / cfg.max_tokens_per_project
                if used_pct >= cfg.warn_at_pct:
                    warnings.append(f"project usage at {used_pct:.0f}% of limit")

        # per day
        if cfg.max_tokens_per_day:
            since = _start_of_day()
            totals = self.storage.budget_totals(self.storage.project, since=since)
            if totals["tokens"] + estimated_tokens > cfg.max_tokens_per_day:
                return BudgetDecision(
                    False,
                    f"daily limit {cfg.max_tokens_per_day} tokens would be exceeded "
                    f"(used {totals['tokens']})",
                    max(0, cfg.max_tokens_per_day - totals["tokens"]),
                    0.0, warnings,
                )
            if cfg.warn_at_pct:
                used_pct = 100.0 * (totals["tokens"] + estimated_tokens) / cfg.max_tokens_per_day
                if used_pct >= cfg.warn_at_pct:
                    warnings.append(f"daily usage at {used_pct:.0f}% of limit")

        # cost (no hard limit in config, but track)
        remaining_tokens = 0
        if cfg.max_tokens_per_project:
            remaining_tokens = max(0, cfg.max_tokens_per_project - (self.storage.budget_totals(self.storage.project)["tokens"] + estimated_tokens))
        elif cfg.max_tokens_per_day:
            remaining_tokens = max(0, cfg.max_tokens_per_day - (self.storage.budget_totals(self.storage.project, since=_start_of_day())["tokens"] + estimated_tokens))

        return BudgetDecision(True, "", remaining_tokens, 0.0, warnings)

    def forecast(self, avg_tokens_per_request: int, n_requests: int) -> dict[str, Any]:
        return {
            "estimated_tokens": avg_tokens_per_request * n_requests,
            "requests": n_requests,
        }

    def _session_tokens(self, session_id: str) -> int:
        # sessions are tracked in requests table by project; approximate by recent window
        since = time.time() - 3600 * 8
        totals = self.storage.budget_totals(self.storage.project, since=since)
        return totals["tokens"]


def _start_of_day() -> float:
    now = time.time()
    # midnight UTC
    import datetime

    dt = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.timestamp()
