# Changelog

## [0.1.0] - 2026-11-01

### Added
- Core library: config, tokens, protect, compress, context, storage, cache,
  budget, router, metrics, optimizer.
- OpenAI-compatible gateway with streaming passthrough.
- CLI: serve, stats, compress, context, cache, budget, models, optimize, init, config.
- Eval harness (dry-run + live).
- 45 tests.

### Notes
- Dependency-free core; tiktoken optional; semantic cache uses a local
  hashing embedder.
