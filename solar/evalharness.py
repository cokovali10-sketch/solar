"""Evaluation harness for SOLAR.

Tasks are JSON or JSONL. Each task:

{
  "id": "unique",
  "messages": [{"role": "user", "content": "..."}],
  "checks": {
    "max_tokens": 200,
    "must_contain": ["substring"],
    "must_not_contain": ["substring"],
    "min_savings_pct": 10.0,
    "max_cost_usd": 0.01
  }
}

Dry run measures compression only. Live mode calls an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import urllib.request

from .config import SolarConfig
from .optimizer import SolarOptimizer


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    return [data]


def _check_text(checks: dict[str, Any], text: str) -> list[str]:
    failures: list[str] = []
    for needle in checks.get("must_contain", []):
        if needle not in text:
            failures.append(f"missing: {needle!r}")
    for needle in checks.get("must_not_contain", []):
        if needle in text:
            failures.append(f"forbidden present: {needle!r}")
    return failures


def run_dry(tasks: list[dict], config: SolarConfig) -> dict[str, Any]:
    optimizer = SolarOptimizer(config)
    results = []
    passed = 0
    for task in tasks:
        messages = task.get("messages", [])
        optimized = optimizer.prepare(messages)
        checks = task.get("checks", {})
        failures: list[str] = []
        if "max_tokens" in checks and optimized.report.optimized_tokens > checks["max_tokens"]:
            failures.append(
                f"optimized_tokens {optimized.report.optimized_tokens} > {checks['max_tokens']}"
            )
        if "min_savings_pct" in checks and optimized.report.savings_pct < checks["min_savings_pct"]:
            failures.append(
                f"savings {optimized.report.savings_pct}% < {checks['min_savings_pct']}%"
            )
        # integrity is a hard requirement
        if not optimized.report.integrity_ok:
            failures.append("integrity check failed")
        ok = not failures
        if ok:
            passed += 1
        results.append({
            "id": task.get("id", "?"),
            "ok": ok,
            "failures": failures,
            "original_tokens": optimized.report.original_tokens,
            "optimized_tokens": optimized.report.optimized_tokens,
            "savings_pct": optimized.report.savings_pct,
            "integrity_ok": optimized.report.integrity_ok,
            "cache_hit": optimized.report.cache_hit,
        })
    optimizer.close()
    return {
        "mode": "dry-run",
        "total": len(tasks),
        "passed": passed,
        "pass_rate": (100.0 * passed / len(tasks)) if tasks else 0.0,
        "results": results,
    }


def run_live(tasks: list[dict], config: SolarConfig, base_url: str, api_key: str = "") -> dict[str, Any]:
    optimizer = SolarOptimizer(config)
    results = []
    passed = 0
    for task in tasks:
        messages = task.get("messages", [])
        optimized = optimizer.prepare(messages)
        checks = task.get("checks", {})
        # budget check
        start = time.time()
        payload = {
            "model": optimized.model,
            "messages": [m.as_dict() for m in optimized.messages],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            latency_ms = int((time.time() - start) * 1000)
            choices = response.get("choices") or []
            content = ""
            if choices:
                content = (choices[0].get("message") or {}).get("content", "") or ""
            usage = response.get("usage") or {}
            cost = 0.0
            info = optimizer.router.model_info(optimized.model)
            if info:
                from .tokens import cost_usd

                cost = cost_usd(
                    int(usage.get("prompt_tokens", 0)),
                    int(usage.get("completion_tokens", 0)),
                    info.get("cost_in", 0.0),
                    info.get("cost_out", 0.0),
                )
            failures = _check_text(checks, content)
            if "max_cost_usd" in checks and cost > checks["max_cost_usd"]:
                failures.append(f"cost {cost} > {checks['max_cost_usd']}")
            if "max_tokens" in checks:
                total = int(usage.get("total_tokens", 0))
                if total and total > checks["max_tokens"]:
                    failures.append(f"total_tokens {total} > {checks['max_tokens']}")
            ok = not failures
            if ok:
                passed += 1
            results.append({
                "id": task.get("id", "?"),
                "ok": ok,
                "failures": failures,
                "latency_ms": latency_ms,
                "cost_usd": cost,
                "savings_pct": optimized.report.savings_pct,
                "content_preview": content[:200],
            })
        except Exception as e:
            results.append({"id": task.get("id", "?"), "ok": False, "failures": [str(e)]})
    optimizer.close()
    return {
        "mode": "live",
        "total": len(tasks),
        "passed": passed,
        "pass_rate": (100.0 * passed / len(tasks)) if tasks else 0.0,
        "results": results,
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="solar-eval")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8420/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = SolarConfig.load(args.config)
    tasks = load_tasks(args.tasks)
    if args.mode == "live":
        report = run_live(tasks, config, args.base_url, args.api_key)
    else:
        report = run_dry(tasks, config)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
