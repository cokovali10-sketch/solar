# SOLAR — Token Optimization Layer for LLM Workflows

SOLAR sits between you and any OpenAI-compatible LLM endpoint and makes your
tokens go further. It compresses prompts without touching code, selects only
the relevant project files, caches answers to repeated questions, routes each
task to the cheapest model that can handle it, and enforces token budgets.

The honest goal: **spend fewer tokens without making the model wrong.**

```
user prompt ──► SOLAR ──► upstream LLM
                │
                ├── protect code/ids/paths/numbers
                ├── compress prose
                ├── inject relevant context (budgeted)
                ├── semantic cache lookup
                ├── model routing
                └── budget check
```

## Status

Version 0.1.0 — MVP. Core library, CLI and OpenAI-compatible gateway are
working. 45 tests pass.

## Installation

```bash
cd D:\PortfolioProjects\SOLAR
pip install -e .
# optional exact tokenizer:
pip install -e ".[exact]"
```

## Quick start

```bash
# 1. create a config
solar init

# 2. inspect resolved config
solar config

# 3. compress a prompt from stdin
echo "Please could you kindly refactor load_config() ..." | solar compress --verbose

# 4. build a context bundle
solar context "refactor load_config" --root . --json

# 5. run the gateway
solar serve
# then point your client at http://127.0.0.1:8420/v1
```

## What it does

### A. Prompt compression (`solar/compress.py`)

- Removes polite filler ("please", "could you", "I would like you to", ...).
- Replaces verbose phrases ("in order to" → "to", "due to the fact that" → "because").
- Removes soft filler words ("just", "simply", "really", "basically", ...).
- Deduplicates repeated sentences and paragraphs.
- Drops conversational meta ("Hope this helps", "Let me know if...") in answers.
- Collapses repeated log lines into `[xN] signature` form.

**Hard rule:** code fences, inline code, identifiers, paths, URLs, numbers,
quoted literals and shell commands are never rewritten. They are split out
before compression and verified afterwards. If verification fails, the original
text is returned untouched.

### B. Context optimizer (`solar/context.py`)

- Walks the project, ranks files by relevance to the query (path + content
  overlap, changed-files bonus, test-file bonus).
- Extracts the best line window per file, within a token budget.
- Includes dependency summaries (public defs of imported modules).
- Adds a compact project structure tree and the git diff.
- Falls back to recently modified files when not a git repo.

### C. Semantic cache (`solar/cache.py`)

- Exact-match hash first, then cosine similarity on a local hashing embedder
  (no external service, no API key).
- Two modes: `safe` (only reuse when the project context fingerprint still
  matches and the entry is verified) and `semantic` (similarity only).
- TTL, LRU eviction, per-project/model scoping.

### D. Smart model router (`solar/router.py`)

- Detects task kind (simple / docs / code / debug / architecture) from the
  prompt.
- Picks the cheapest model that meets the required quality tier and fits the
  context window.
- User-defined rules (`when` regex → model) take precedence.
- Honours `SOLAR_ROUTER=1` to enable; disabled by default.

### E. Token budget manager (`solar/budget.py`)

- Per request, per session, per project and per day limits.
- Warnings at a configurable percentage (default 80%).
- Forecasts and remaining-token reporting.

### F. Metrics (`solar/metrics.py`)

- Per project / model / day aggregates: tokens, savings, cost, cache hit rate.
- CLI: `solar stats`.

### G. Gateway (`solar/gateway.py`)

- OpenAI-compatible: `POST /v1/chat/completions`, `GET /v1/models`.
- Streaming passthrough (SSE) with usage capture.
- Endpoints: `/health`, `/solar/stats`, `/solar/requests`, `/solar/cache`,
  `/solar/budget`, `/solar/compress`, `/solar/context`, `/solar/cache/clear`.
- Adds `X-Solar-Saved-Tokens`, `X-Solar-Savings-Pct`, `X-Solar-Cache`,
  `X-Solar-Model` headers.

## Configuration

`solar.json` (created by `solar init`):

```json
{
  "project": "default",
  "model": "gpt-4o-mini",
  "budget": { "max_tokens_per_request": 0, "max_tokens_per_session": 0,
              "max_tokens_per_project": 0, "max_tokens_per_day": 0, "warn_at_pct": 80 },
  "cache": { "enabled": true, "similarity_threshold": 0.92,
             "ttl_seconds": 604800, "max_entries": 5000, "mode": "safe" },
  "compression": { "level": "balanced", "min_savings_pct": 5.0,
                   "min_keep_ratio": 0.3, "compress_logs": true, "max_log_lines": 40 },
  "context": { "enabled": true, "max_files": 6, "max_snippet_lines": 60,
               "budget_tokens": 1500, "include_git_diff": true },
  "router": { "enabled": false, "default_model": "gpt-4o-mini" }
}
```

Environment variables override file values: `SOLAR_DB`, `SOLAR_PROJECT`,
`SOLAR_MODEL`, `SOLAR_PORT`, `SOLAR_HOST`, `OPENAI_BASE_URL`, `OLLAMA_BASE_URL`,
`SOLAR_API_KEY`, `SOLAR_CACHE_MODE`, `SOLAR_COMPRESSION_LEVEL`, `SOLAR_ROUTER`.

## Evaluation

```bash
# dry run (compression only, no network)
python -m solar.evalharness --tasks tasks.json --mode dry-run

# live (hits a running gateway or upstream)
python -m solar.evalharness --tasks tasks.json --mode live --base-url http://127.0.0.1:8420/v1
```

## Tests

```bash
python -m pytest tests -q
# 45 passed
```

## Honest limitations

- Compression only helps on prose. Code, identifiers, paths, numbers and
  quoted literals are never touched — that is the point, but it caps savings
  on code-heavy prompts.
- The semantic cache uses a hashed bag-of-words embedder, not a real sentence
  model. False positives are possible on short prompts; `mode="safe"` guards
  against them by requiring a matching context fingerprint.
- Context injection is additive: on a tiny prompt with no project, it will not
  reduce tokens. The headline savings come from compressing verbose prompts and
  caching repeated questions.
- No multi-tenant isolation, no GUI yet.
- Token counts are estimates unless `tiktoken` is installed (`pip install
  solar-tokens[exact]`); the heuristic over-estimates slightly on purpose.

## Architecture

```
solar/
  config.py      configuration (dataclasses, env + json)
  models.py      Message, Usage, OptimizationReport, OptimizedRequest
  tokens.py      token estimation (tiktoken optional, heuristic fallback)
  protect.py     protected-span splitter + integrity check
  compress.py    prose/log/response compression
  context.py     project context extraction
  storage.py     SQLite (requests, cache, sessions, kv)
  cache.py       semantic cache + hashing embedder
  budget.py      budget manager
  router.py      model router
  metrics.py     aggregation + formatting
  optimizer.py   orchestrates the pipeline
  gateway.py     OpenAI-compatible HTTP proxy
  cli.py         command line
  evalharness.py evaluation runner
```

## Integration with FORGE / MATRIX

SOLAR is designed to be the efficiency layer underneath FORGE and MATRIX:

- FORGE sends tasks; SOLAR optimizes the context and picks the model.
- MATRIX provides knowledge and memory; SOLAR caches the results.
- Qorvex provides the sandbox; MATRIX reuses it for simulation.

Shared contract (planned):

```
Task { id, repo, goal, constraints, budget }
ContextBundle { files[], diff, knowledge[], memory[] }
ModelPolicy { model, max_tokens, fallback }
Result { diff, tests, tokens, cost, verdict }
```

## License

MIT
