# Coding Style

Style is enforced by tooling, not by this document: `ruff` for formatting and
linting, `mypy` for types, both wired into `.pre-commit-config.yaml` and run in
CI. Run

```sh
pre-commit run --all-files
```

before pushing, and the lint job will pass.

The settings that are a choice rather than a default live in `pyproject.toml`:

- **Docstrings follow the Google convention** (`[tool.ruff.lint.pydocstyle]`),
  not numpy. Public modules and packages are exempt from the docstring rule.
- Line length 100, double quotes, `py310` target.
- `mypy` runs against `src` only, in the project environment via `uv`, so it
  sees the real types from aiida-core, numpy and pymatgen.

Conventions the linter cannot check — units, estimators, provenance rules, the
FEFF version this plugin targets — are in [`AGENTS.md`](AGENTS.md).
