# Contributing to SOLAR

Thanks for your interest. SOLAR is a small, dependency-light project; please
keep changes focused and tested.

## Development setup

```bash
git clone https://github.com/cokovali10-sketch/solar
cd solar
pip install -e ".[dev]"
python -m pytest tests -q
```

## Guidelines

- **Protected spans are sacred.** Never modify code/identifiers/paths/numbers/
  quotes in compression. If you add a new compression step, add an integrity
  check.
- Keep the core dependency-free. Optional heavy deps go behind extras.
- Add tests for new behavior. Aim for the existing style (pytest, tmp_path).
- Update README and docs/PLAN.md when adding user-facing features.
- Be honest in commit messages and PR descriptions about limitations.

## Commit style

Conventional commits preferred: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.

## Reporting issues

Use the issue templates. Include reproduction steps and environment.
