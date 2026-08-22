set shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    @just --list

install:
    uv sync --extra dev --extra notebook

lint:
    uv run ruff check .

test:
    uv run pytest -q

test-python:
    uv run python -m pytest -q

train:
    uv run python -m src.ml_project.modeling.train

api:
    uv run uvicorn src.api.api:app --host 0.0.0.0 --port 8000

drift reference current:
    uv run python -m src.ml_project.monitoring {{reference}} {{current}}

docker-build:
    docker compose build

docker-smoke:
    docker compose run --rm train python -m src.ml_project.pipeline prepare
    docker compose run --rm -e MLFLOW_ENABLE_MODEL_REGISTRY=false train python -m src.ml_project.modeling.train

# Inicializa o DVC no projeto (caso ainda não tenha sido feito)
dvc-init:
    uv run dvc init

# Puxa os dados/modelos do storage remoto configurado
pull:
    uv run dvc pull

# Envia os dados/modelos alterados para o storage remoto
push:
    uv run dvc push

# Adiciona um novo arquivo ou pasta para rastreio do DVC
track path:
    uv run dvc add {{path}}
    git add {{path}}.dvc .gitignore

dvc-repro:
    uv run dvc repro

serve-docs:
    uv run mkdocs serve
