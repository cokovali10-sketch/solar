# SOLAR Development Plan

## Phase 1 — MVP (done)

- [x] Config (json + env)
- [x] Token estimation (tiktoken optional, heuristic fallback)
- [x] Protection layer (code/ids/paths/numbers/quotes never rewritten)
- [x] Prompt compression (filler, phrases, dedup, logs)
- [x] Context optimizer (ranked files, budget, diff, structure)
- [x] Semantic cache (exact + cosine, safe/semantic modes)
- [x] Model router (task detection, quality tiers, rules)
- [x] Budget manager (request/session/project/day)
- [x] Metrics engine
- [x] Optimizer orchestration
- [x] Gateway (OpenAI-compatible, streaming)
- [x] CLI
- [x] Eval harness (dry-run + live)
- [x] Tests (45 passing)

## Phase 2 — Advanced

- [ ] Response compression wired into gateway (opt-in, lossless-ish)
- [ ] Project digest cached in SQLite (invalidate on file hash change)
- [ ] Better embedder (optional sentence-transformers behind extra)
- [ ] GUI dashboard (tokens before/after, savings, cache, budget)
- [ ] Comparison mode: same task with/without optimization, report deltas
- [ ] Secret scanning before sending code to cloud (reuse ApexGuard patterns)
- [ ] Cost forecasting and alerts

## Phase 3 — MATRIX integration

- [ ] Tool registry (JSON Schema + examples + safety rules)
- [ ] Task set with tests (30+ real repo tasks)
- [ ] Docker runner + git worktree (reuse Qorvex sandbox)
- [ ] Memory of confirmed fixes (append-only + index)
- [ ] Scoring: pass/fail, diff size, attempts, tokens

## Phase 4 — FORGE integration

- [ ] Shared contract (Task, ContextBundle, ModelPolicy, Result)
- [ ] Unified agent manager
- [ ] Single dashboard

## Success criteria

SOLAR:
- >= 25% token reduction on a fixed 20-task set with <= 2 pp quality drop.
- Cache hit rate measurable and reported.
- Router picks a cheaper model without quality regression on >= 80% of tasks.

MATRIX:
- >= 60% of tasks pass on first/second attempt.
- No regression on repeated tasks.
