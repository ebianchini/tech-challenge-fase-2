default:
    @just --list

install:
    uv sync --extra dev --extra notebook

lint:
    uv run ruff check .

test:
    uv run pytest -q

train:
    uv run python -m src.ml_project.train

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
