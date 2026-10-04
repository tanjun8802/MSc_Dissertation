# Contributing

Use Python 3.12 and `uv sync --locked --group dev`. Install the `notebooks` extra
for interactive work. Work on a branch and open a pull request.

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/check_notebooks.py
uv run python -m build
```

For training mathematics changes, run the migration equivalence script when
applicable and explain intended differences. Tests should exercise real updates,
replay semantics, gradient isolation or other meaningful contracts. Long runs
should record configuration, seeds, task order, interaction/update budget,
hardware and output artifact location.

Keep settings in `configs/`, implementations in `src/tbtrl`, and notebooks as
small clients. New environments must meet the [environment contract](docs/environments.md).
Avoid global mutable experiment state and machine-specific paths. Preserve
failed runs in aggregation.

Do not commit datasets, checkpoints, PDFs, videos or notebook outputs. Publish
large research artifacts separately with their configuration identifiers.
Before a formal open-source release, the owner should select a licence and
confirm attribution for retained code and research assets.
