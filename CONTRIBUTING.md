# Contributing to SmartMemo

Thanks for your interest in SmartMemo. This guide covers local development, the quality
gates, and how releases are made.

## Development setup

SmartMemo uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management. Python 3.11–3.14 are supported.

```bash
uv sync --all-extras
```

The `[ml]` extra (PyTorch, FAISS, SentenceTransformers) is required to import `smartmemo`
itself, so `--all-extras` is the normal development sync.

## Quality gates

These four commands are exactly what CI runs. All must pass before a change merges:

```bash
uv run ruff format --check   # formatting
uv run ruff check            # lint
uv run pyright               # type checking
uv run pytest                # tests
```

CI runs them on Python 3.11, 3.12, 3.13, and 3.14.

## Concurrency model

The SQLite store opens the database in WAL mode with a 5-second busy timeout. A single
store instance is safe to use from multiple threads of one process — writes are
serialized by an internal re-entrant lock, and the connection is created with
`check_same_thread=False`. SmartMemo is not a distributed cache: multiple processes
writing the same database file rely solely on SQLite's file locking, and heavy
multi-process write contention can still raise `sqlite3.OperationalError`.

## The bundled classifier

The package ships a pretrained equivalence classifier at
`src/smartmemo/_models/classifier-v2.pt`, force-included into the wheel by the
`[tool.hatch.build.targets.wheel.force-include]` table in `pyproject.toml`.

It is reproducible from committed data:

```bash
# Optional: regenerate the dataset from the prompt corpus (requires a local Ollama model)
python scripts/generate_training_data.py

# Train and package the classifier from the committed dataset
uv run python scripts/train_classifier.py
```

`train_classifier.py` writes the checkpoint plus an auditable model card
(`classifier-v2.report.json`) next to it. SmartMemo is precision-first: the acceptance
gate requires the classifier to beat the cosine baseline by at least 10 precision points
at equal recall on the held-out gold set. See `docs/ml/` for details.

## Release process

Releases are tag-triggered:

1. Bump `version` in `pyproject.toml`.
2. Add a new section to `CHANGELOG.md` describing what shipped.
3. Commit the changes to `main`.
4. Tag the commit `vX.Y.Z` and push the tag.

Pushing the tag triggers `.github/workflows/publish-pypi.yml`, which runs the full test
suite before building and publishing to PyPI — a broken tag cannot publish.
