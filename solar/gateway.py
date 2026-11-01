"""OpenAI-compatible gateway for SOLAR."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import SolarConfig
from .metrics import MetricsEngine
from .models import Message, Usage
from .optimizer import SolarOptimizer
from .tokens import messages_tokens


class SolarGateway:
    def __init__(self, config: SolarConfig, optimizer: SolarOptimizer | None = None):
        self.config = config
        self.optimizer = optimizer or SolarOptimizer(config)
        self.metrics = MetricsEngine(self.optimizer.storage)

    def upstream_base_url(self, model: str) -> str:
        cfg = self.config.gateway
        info = self.optimizer.router.model_info(model)
        if info and info.get("provider") == "ollama":
            return cfg.ollama_base_url
        if model.startswith("ollama") or ":" in model and model.split(":")[0] in {"llama", "qwen", "mistral", "codellama"}:
            # heuristic: model names with a colon are often local Ollama tags
            if "gpt" not in model and "claude" not in model:
                return cfg.ollama_base_url
        return cfg.upstream_base_url

    def make_handler(self):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                # quiet by default; uncomment for debugging
                pass

            # ---------------- helpers
            def _send_json(self, status: int, payload: Any, headers: dict | None = None):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                if headers:
                    for key, value in headers.items():
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def _read_body(self) -> dict:
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length) if length > 0 else b""
                if not raw:
                    return {}
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    return {"_raw": raw.decode("utf-8", errors="replace")}

            def _auth_ok(self) -> bool:
                expected = gateway.config.gateway.api_key
                if not expected:
                    return True
                auth = self.headers.get("Authorization", "")
                token = auth[len("Bearer ") :] if auth.lower().startswith("bearer ") else ""
                return token == expected

            def _upstream_headers(self) -> dict:
                headers = {"Content-Type": "application/json"}
                auth = self.headers.get("Authorization")
                if auth:
                    headers["Authorization"] = auth
                elif gateway.config.gateway.api_key:
                    # fall back to configured key for upstream if none provided
                    pass
                api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SOLAR_UPSTREAM_API_KEY")
                if api_key and "Authorization" not in headers:
                    headers["Authorization"] = f"Bearer {api_key}"
                return headers

            # ---------------- routing
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/health" or path == "/api/v1/health":
                    self._send_json(200, {"status": "ok", "service": "solar", "version": "0.1.0"})
                    return
                if path == "/v1/models":
                    models = [
                        {
                            "id": m["name"],
                            "object": "model",
                            "owned_by": m.get("provider", "solar"),
                        }
                        for m in gateway.config.models
                    ]
                    self._send_json(200, {"object": "list", "data": models})
                    return
                if path == "/solar/stats":
                    summary = gateway.metrics.summary()
                    self._send_json(200, summary)
                    return
                if path == "/solar/requests":
                    rows = gateway.metrics.recent(limit=50)
                    self._send_json(200, {"requests": rows})
                    return
                if path == "/solar/cache":
                    rows = gateway.optimizer.cache.list_entries()
                    self._send_json(200, {"entries": rows})
                    return
                if path == "/solar/budget":
                    totals = gateway.optimizer.storage.budget_totals(gateway.config.project)
                    self._send_json(200, {"totals": totals, "config": gateway.config.budget.as_dict() if hasattr(gateway.config.budget, "as_dict") else {}})
                    return
                self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                if not self._auth_ok():
                    self._send_json(401, {"error": {"message": "unauthorized", "type": "auth"}})
                    return

                if path == "/v1/chat/completions":
                    self._handle_chat()
                    return
                if path == "/solar/compress":
                    body = self._read_body()
                    text = body.get("text", "")
                    level = body.get("level")
                    from .compress import compress_text

                    result = compress_text(
                        text,
                        model=body.get("model", gateway.config.model),
                        config=gateway.config.compression,
                        level=level,
                        role=body.get("role", "user"),
                    )
                    self._send_json(200, result.as_dict() if hasattr(result, "as_dict") else {
                        "original": result.original,
                        "compressed": result.compressed,
                        "original_tokens": result.original_tokens,
                        "compressed_tokens": result.compressed_tokens,
                        "savings_pct": result.savings_pct,
                        "integrity_ok": result.integrity_ok,
                        "notes": result.notes,
                    })
                    return
                if path == "/solar/context":
                    body = self._read_body()
                    root = body.get("root") or gateway.optimizer.project_root
                    from .context import build_context

                    bundle = build_context(root, body.get("query", ""), gateway.config.context, body.get("model", gateway.config.model))
                    self._send_json(200, bundle.as_dict())
                    return
                if path == "/solar/cache/clear":
                    removed = gateway.optimizer.cache.clear()
                    self._send_json(200, {"removed": removed})
                    return
                if path == "/solar/budget/check":
                    body = self._read_body()
                    decision = gateway.optimizer.check_budget(
                        int(body.get("tokens", 0)), float(body.get("cost", 0.0))
                    )
                    self._send_json(200, {
                        "allowed": decision.allowed,
                        "reason": decision.reason,
                        "remaining_tokens": decision.remaining_tokens,
                        "warnings": decision.warnings,
                    })
                    return
                self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

            def _handle_chat(self):
                body = self._read_body()
                messages = body.get("messages", [])
                if not isinstance(messages, list):
                    self._send_json(400, {"error": {"message": "messages must be a list"}})
                    return
                model_override = body.get("model")
                stream = bool(body.get("stream", False))

                start = time.time()
                optimized = gateway.optimizer.prepare(
                    messages,
                    model=model_override,
                    use_context=gateway.config.context.enabled,
                    use_cache=gateway.config.cache.enabled,
                    use_compression=True,
                )

                report = optimized.report
                cached = optimized.cached_response

                # budget check
                estimated = optimized.report.optimized_tokens
                decision = gateway.optimizer.check_budget(estimated)
                if not decision.allowed:
                    self._send_json(429, {"error": {"message": decision.reason, "type": "budget"}})
                    return

                extra_headers = {}
                if gateway.config.gateway.add_report_header:
                    extra_headers["X-Solar-Saved-Tokens"] = str(report.saved_tokens)
                    extra_headers["X-Solar-Savings-Pct"] = str(report.savings_pct)
                    extra_headers["X-Solar-Cache"] = "hit" if report.cache_hit else "miss"
                    extra_headers["X-Solar-Model"] = optimized.model

                # cache hit: return cached response (non-stream only)
                if cached is not None and not stream:
                    # attach report
                    cached = dict(cached)
                    cached.setdefault("solar", report.as_dict())
                    self._send_json(200, cached, headers=extra_headers)
                    gateway.optimizer.record(report, latency_ms=int((time.time() - start) * 1000))
                    return

                # forward to upstream
                upstream_body = {
                    "model": optimized.model,
                    "messages": [m.as_dict() for m in optimized.messages],
                }
                # pass through common fields
                for field in ("temperature", "top_p", "n", "stop", "max_tokens", "max_completion_tokens", "presence_penalty", "frequency_penalty", "seed", "tools", "tool_choice", "response_format", "stream_options"):
                    if field in body:
                        upstream_body[field] = body[field]
                if stream:
                    upstream_body["stream"] = True
                    if "stream_options" not in upstream_body:
                        upstream_body["stream_options"] = {"include_usage": True}

                base_url = gateway.upstream_base_url(optimized.model).rstrip("/")
                url = base_url + "/chat/completions"
                headers = gateway._upstream_headers()
                data = json.dumps(upstream_body, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")

                try:
                    with urllib.request.urlopen(req, timeout=gateway.config.gateway.request_timeout) as resp:
                        if stream:
                            self.send_response(resp.status)
                            self.send_header("Content-Type", resp.headers.get("Content-Type", "text/event-stream"))
                            for key, value in extra_headers.items():
                                self.send_header(key, value)
                            self.send_header("Transfer-Encoding", "chunked")
                            self.end_headers()
                            usage = Usage()
                            model_used = optimized.model
                            for raw in resp:
                                self.wfile.write(raw)
                                self.wfile.flush()
                                # try parse SSE usage
                                text = raw.decode("utf-8", errors="replace")
                                for line in text.splitlines():
                                    if not line.startswith("data:"):
                                        continue
                                    payload = line[5:].strip()
                                    if payload == "[DONE]":
                                        continue
                                    try:
                                        chunk = json.loads(payload)
                                    except Exception:
                                        continue
                                    if chunk.get("model"):
                                        model_used = chunk["model"]
                                    u = chunk.get("usage")
                                    if u:
                                        usage = Usage.from_api(u)
                            # record
                            if usage.total_tokens == 0:
                                usage.prompt_tokens = messages_tokens(optimized.messages, optimized.model)
                            gateway.optimizer.record(
                                report,
                                usage=usage,
                                latency_ms=int((time.time() - start) * 1000),
                            )
                            # store cache (unverified by default)
                            try:
                                gateway.optimizer.store_cache(
                                    messages=optimized.messages,
                                    model=optimized.model,
                                    response={"note": "streamed; body not cached"},
                                    context_fingerprint="",
                                    verified=False,
                                )
                            except Exception:
                                pass
                            return
                        else:
                            raw = resp.read()
                            status = resp.status
                except urllib.error.HTTPError as e:
                    raw = e.read()
                    status = e.code
                except Exception as e:
                    self._send_json(502, {"error": {"message": str(e), "type": "upstream"}})
                    return

                # non-stream response handling
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                usage = Usage.from_api(payload.get("usage"))
                if usage.total_tokens == 0:
                    # estimate completion tokens from content
                    content = ""
                    choices = payload.get("choices") or []
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content", "") or ""
                    usage.completion_tokens = estimate_tokens(content, optimized.model)
                    usage.prompt_tokens = messages_tokens(optimized.messages, optimized.model)
                    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
                # cost
                info = gateway.optimizer.router.model_info(optimized.model)
                if info:
                    from .tokens import cost_usd

                    usage.cost_usd = cost_usd(
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        info.get("cost_in", 0.0),
                        info.get("cost_out", 0.0),
                    )

                payload.setdefault("solar", report.as_dict())
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body_bytes)))
                for key, value in extra_headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body_bytes)

                gateway.optimizer.record(
                    report,
                    usage=usage,
                    latency_ms=int((time.time() - start) * 1000),
                )

                # store cache (only if no tool calls and success)
                try:
                    choices = payload.get("choices") or []
                    has_tool_calls = False
                    if choices:
                        msg = choices[0].get("message", {}) or {}
                        has_tool_calls = bool(msg.get("tool_calls"))
                    if status < 400 and not has_tool_calls:
                        gateway.optimizer.store_cache(
                            messages=optimized.messages,
                            model=optimized.model,
                            response=payload,
                            context_fingerprint="",
                            verified=False,
                        )
                except Exception:
                    pass

        return Handler

    def serve(self) -> None:
        handler = self.make_handler()
        httpd = ThreadingHTTPServer((self.config.gateway.host, self.config.gateway.port), handler)
        print(f"SOLAR gateway listening on http://{self.config.gateway.host}:{self.config.gateway.port}")
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
            self.optimizer.close()
