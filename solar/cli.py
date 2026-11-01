"""Command-line interface for SOLAR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .budget import BudgetManager
from .cache import Cache
from .compress import compress_text
from .config import SolarConfig
from .context import build_context
from .gateway import SolarGateway
from .metrics import MetricsEngine
from .models import Message
from .optimizer import SolarOptimizer
from .router import ModelRouter
from .storage import Storage
from .tokens import estimate_tokens, messages_tokens


def _load_config(args) -> SolarConfig:
    path = getattr(args, "config", None)
    return SolarConfig.load(path)


def _read_text_arg(args) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return getattr(args, "text", "") or ""


def cmd_serve(args) -> int:
    config = _load_config(args)
    gateway = SolarGateway(config)
    gateway.serve()
    return 0


def cmd_stats(args) -> int:
    config = _load_config(args)
    storage = Storage(config.db_path, project=config.project)
    metrics = MetricsEngine(storage)
    summary = metrics.summary(days=args.days)
    print(metrics.format_summary(summary))
    if args.recent:
        print()
        print(metrics.format_table(metrics.recent(limit=args.recent)))
    storage.close()
    return 0


def cmd_compress(args) -> int:
    config = _load_config(args)
    text = _read_text_arg(args)
    result = compress_text(
        text,
        model=args.model or config.model,
        config=config.compression,
        level=args.level,
        role=args.role,
    )
    if args.json:
        print(json.dumps({
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "savings_pct": result.savings_pct,
            "integrity_ok": result.integrity_ok,
            "notes": result.notes,
            "compressed": result.compressed,
        }, ensure_ascii=False, indent=2))
    else:
        print(result.compressed)
        if args.verbose:
            print(f"\n# {result.original_tokens} -> {result.compressed_tokens} tokens ({result.savings_pct}% saved)", file=sys.stderr)
            if result.notes:
                print("# " + "; ".join(result.notes), file=sys.stderr)
    return 0


def cmd_context(args) -> int:
    config = _load_config(args)
    root = args.root or "."
    bundle = build_context(root, args.query, config.context, args.model or config.model)
    if args.json:
        print(json.dumps(bundle.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(bundle.text)
        print(f"\n# {bundle.tokens} tokens, {len(bundle.files)} files", file=sys.stderr)
    return 0


def cmd_cache(args) -> int:
    config = _load_config(args)
    storage = Storage(config.db_path, project=config.project)
    cache = Cache(storage, config.cache)
    action = args.action
    if action == "list":
        rows = cache.list_entries(limit=args.limit)
        for row in rows:
            print(f"{row['cache_key'][:12]}  {row['model']:<20}  hits={row['hits']}  verified={row['verified']}  {row['normalized_prompt'][:60]}")
    elif action == "clear":
        removed = cache.clear()
        print(f"removed {removed} entries")
    elif action == "mark":
        cache.mark_verified(args.key, True)
        print("marked verified")
    storage.close()
    return 0


def cmd_budget(args) -> int:
    config = _load_config(args)
    storage = Storage(config.db_path, project=config.project)
    budget = BudgetManager(storage, config.budget)
    decision = budget.check(int(args.tokens), float(args.cost))
    print(json.dumps({
        "allowed": decision.allowed,
        "reason": decision.reason,
        "remaining_tokens": decision.remaining_tokens,
        "warnings": decision.warnings,
    }, indent=2))
    storage.close()
    return 0


def cmd_models(args) -> int:
    config = _load_config(args)
    for model in config.models:
        print(f"{model['name']:<24} {model.get('provider',''):<10} q={model.get('quality')} ctx={model.get('context_window')} in=${model.get('cost_in')} out=${model.get('cost_out')}")
    return 0


def cmd_optimize(args) -> int:
    config = _load_config(args)
    storage = Storage(config.db_path, project=config.project)
    optimizer = SolarOptimizer(config, storage=storage, project_root=args.root)
    text = _read_text_arg(args)
    messages = [{"role": "user", "content": text}]
    optimized = optimizer.prepare(messages, model=args.model, use_context=not args.no_context)
    print(json.dumps({
        "model": optimized.model,
        "original_tokens": optimized.report.original_tokens,
        "optimized_tokens": optimized.report.optimized_tokens,
        "saved_tokens": optimized.report.saved_tokens,
        "savings_pct": optimized.report.savings_pct,
        "cache_hit": optimized.report.cache_hit,
        "routed_model": optimized.report.routed_model,
        "routing_reason": optimized.report.routing_reason,
        "integrity_ok": optimized.report.integrity_ok,
        "notes": optimized.report.notes,
        "messages": [m.as_dict() for m in optimized.messages],
    }, ensure_ascii=False, indent=2))
    storage.close()
    return 0


def cmd_init(args) -> int:
    path = Path(args.path or "solar.json")
    config = SolarConfig()
    config.save(path)
    print(f"wrote {path}")
    return 0


def cmd_config(args) -> int:
    config = _load_config(args)
    print(json.dumps(config.dump(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solar", description="SOLAR token optimizer")
    parser.add_argument("--config", "-c", default=None, help="path to solar.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the OpenAI-compatible gateway")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("stats", help="show usage statistics")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--recent", type=int, default=0)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("compress", help="compress text from stdin or --file")
    p.add_argument("text", nargs="?")
    p.add_argument("--file", "-f")
    p.add_argument("--level", choices=["conservative", "balanced", "aggressive"])
    p.add_argument("--model", default=None)
    p.add_argument("--role", default="user", choices=["user", "assistant", "system"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_compress)

    p = sub.add_parser("context", help="build a context bundle for a query")
    p.add_argument("query")
    p.add_argument("--root", default=".")
    p.add_argument("--model", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("cache", help="manage the semantic cache")
    p.add_argument("action", choices=["list", "clear", "mark"])
    p.add_argument("--key", default=None)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("budget", help="check a token budget")
    p.add_argument("tokens", type=int)
    p.add_argument("--cost", type=float, default=0.0)
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("models", help="list configured models")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("optimize", help="dry-run the full optimization pipeline")
    p.add_argument("text", nargs="?")
    p.add_argument("--file", "-f")
    p.add_argument("--model", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--no-context", action="store_true")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("init", help="write a default solar.json")
    p.add_argument("--path", default="solar.json")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("config", help="print resolved config")
    p.add_argument("show", nargs="?")
    p.set_defaults(func=cmd_config)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
