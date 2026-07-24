# Tech Challenge Fase 2

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
  <img alt="Scikit-Learn" src="https://img.shields.io/badge/scikit--learn-1.5-FFB300?logo=scikit-learn&logoColor=white" />
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-2.x-0194E2?logo=mlflow&logoColor=white" />
  <img alt="DVC" src="https://img.shields.io/badge/DVC-3.x-13ADC7?logo=dvc&logoColor=white" />
  <img alt="Ruff" src="https://img.shields.io/badge/Ruff-0.6-FFB000?logo=ruff&logoColor=white" />
</p>

Projeto base para desenvolvimento de soluções de Machine Learning com Python, Scikit-Learn, MLflow, DVC, Ruff e automação com `just`.

## Objetivo

Estruturar um repositório pronto para experimentação, treinamento e rastreio de modelos em um pipeline reproduzível.

## Estrutura do repositório

```text
.
├── src/
├── tests/
├── data/
├── models/
├── configs/
├── pyproject.toml
├── uv.lock
├── dvc.yaml
├── Dockerfile
└── justfile
```

## Requisitos

- Python 3.11+
- `uv`
- `just`
- Docker (opcional para execução em container)

## Instalação

```bash
uv sync --extra dev
```

> Recomendação: use Python 3.11 a 3.13. O projeto evita o uso do MLflow completo para reduzir o custo de instalação em ambientes Windows.

Para ativar o ambiente virtual local:

```bash
source .venv/bin/activate
```

## Execução rápida

```bash
uv run python -m src.ml_project.train
```

## Pipeline com DVC

```bash
dvc repro
```

## Automação com `just`

```bash
just install
just lint
just test
just train
```

## Variáveis de ambiente

Copie o arquivo `.env.example` para `.env` e ajuste os valores.

## Docker

O `Dockerfile` utiliza multi-stage build, separando a etapa de build da etapa de runtime.

## Contribuição

1. Crie uma branch para sua feature.
2. Execute lint e testes antes do commit.
3. Mantenha o pipeline e o tracking de experimentos reprodutíveis.

## Licença

MIT
