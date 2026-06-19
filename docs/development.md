# Development

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # creates the venv, installs deps + the chessvision package
```

`uv sync` installs the project in editable mode, so source edits take effect
immediately and the `gm-*` console scripts (defined in `pyproject.toml`) are
available via `uv run`.

## Project layout

See [architecture](architecture.md) for the full module map and data flow. In
short:

- `chessvision/core/` — shared library (board detection, occupancy diff, game
  tracking, display helper). No GUI loops; safe to import anywhere.
- `chessvision/app/` — interactive applications.
- `chessvision/training/` — dataset capture, labelling, training, export.
- `chessvision/settings.py` — all configuration (see [configuration](configuration.md)).

Dependency rule: `app` and `training` may import `core`; `core` imports only
`settings`. Keep new shared code in `core`.

## Console scripts

Each entry point in `[project.scripts]` maps a command to a function:

| Command | Target | Notes |
| --- | --- | --- |
| `gm-detect` | `chessvision.app.detect:cli` | main app; `--from-fen`, `--detect`, `--web [PORT]` |
| `gm-view` | `chessvision.app.view_camera:main` | raw stream preview |
| `gm-corners` | `chessvision.core.board:main` | board-detection debug |
| `gm-capture` | `chessvision.training.capture_dataset:main` | |
| `gm-autolabel` | `chessvision.training.autolabel_images:main` | |
| `gm-train` | `chessvision.training.train:main` | |
| `gm-export` | `chessvision.training.export:main` | |

Anything launchable as a command needs a zero-arg entry function (`main()` or
`cli()`); put `argparse` inside it, and keep heavy work (model loading, capture
loops) out of module top-level so workers/imports stay cheap.

## Linting & formatting

[ruff](https://docs.astral.sh/ruff/) is the dev dependency (config in
`pyproject.toml`):

```bash
uv run ruff check .      # lint
uv run ruff check --fix . # autofix
uv run ruff format .     # format
```

The `Makefile` wraps the common commands — run `make help`.

## Adding a setting

1. Add a field (with a comment) to `chessvision/settings.py`.
2. Read it as `settings.<name>` where needed.
3. Document it in [configuration](configuration.md).

It's then overridable as `GRANDMASTER_<NAME>` via env or `.env`. Note: `Literal`
int fields can't be set from env strings (pydantic limitation) — prefer plain
`int`/`float`/`bool`/`str` if env-overridability matters.

## Notebooks

Exploratory Jupyter notebooks live in `jupyter_notebooks/`. They are scratch
space, not part of the pipeline.
