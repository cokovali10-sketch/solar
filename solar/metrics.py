"""Metrics aggregation for SOLAR."""

from __future__ import annotations

from typing import Any

from .storage import Storage


class MetricsEngine:
    def __init__(self, storage: Storage):
        self.storage = storage

    def summary(self, project: str | None = None, days: int = 30) -> dict[str, Any]:
        return self.storage.stats(project=project, days=days)

    def recent(self, limit: int = 20, project: str | None = None) -> list[dict]:
        return self.storage.list_requests(limit=limit, project=project)

    def format_summary(self, summary: dict[str, Any]) -> str:
        lines = [
            f"Project: {summary['project']}",
            f"Window: {summary['days']} days",
            f"Requests: {summary['requests']}",
            f"Original tokens: {summary['original_tokens']}",
            f"Optimized tokens: {summary['optimized_tokens']}",
            f"Saved tokens: {summary['saved_tokens']} ({summary['savings_pct']}%)",
            f"Prompt tokens: {summary['prompt_tokens']}",
            f"Completion tokens: {summary['completion_tokens']}",
            f"Cost USD: {summary['cost_usd']}",
            f"Cache hits: {summary['cache_hits']} ({summary['cache_hit_rate']}%)",
        ]
        if summary.get("by_model"):
            lines.append("By model:")
            for model, data in summary["by_model"].items():
                lines.append(f"  {model}: {data['requests']} req, {data['tokens']} tok, saved {data['saved']}")
        return "\n".join(lines)

    def format_table(self, rows: list[dict]) -> str:
        if not rows:
            return "(no requests)"
        lines = ["id  ts                  model                  orig  opt   saved cache"]
        for row in rows:
            import datetime

            ts = datetime.datetime.fromtimestamp(row["ts"]).strftime("%m-%d %H:%M")
            lines.append(
                f"{row['id']:<3} {ts:<18} {row['model'][:20]:<20} "
                f"{row['original_tokens']:<5} {row['optimized_tokens']:<5} "
                f"{row['saved_tokens']:<5} {'Y' if row['cache_hit'] else 'N'}"
            )
        return "\n".join(lines)
