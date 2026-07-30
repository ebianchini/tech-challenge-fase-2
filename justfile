default:
    @just --list

install:
    uv sync --extra dev

lint:
    uv run ruff check .

test:
    uv run pytest -q

train:
    uv run python -m src.ml_project.train

dvc-repro:
    uv run dvc repro

serve-docs:
    uv run mkdocs serve
