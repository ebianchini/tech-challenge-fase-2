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
python -m src.ml_project.pipeline prepare
python -m src.ml_project.modeling.train
```

## Pipeline com DVC

```bash
dvc repro
```

Na Fase 2, a etapa `prepare` também gera `data/processed/online_shoppers_metadata.json` com:

- fingerprint do dataset;
- distribuição da variável alvo;
- schema esperado e tipos inferidos;
- colunas categóricas, numéricas e features codificadas.

Na Fase 3, o contrato de entrada e saída da inferência está documentado em
[docs/inference-contract.md](docs/inference-contract.md). O contrato define as 17 colunas brutas
obrigatórias, seus tipos e restrições, o payload JSON em lote e o formato de resposta padronizado.
As features derivadas e a coluna `Revenue` são internas ao pipeline e não devem ser enviadas.
A API de inferência foi implementada com FastAPI em `src/ml_project/api.py`, expondo `/health`
e `/predict`, com validação de payload, erros padronizados e logs por requisição.

Para executar a API localmente:

```bash
just api
```

O treino passa a registrar no MLflow:

- benchmark com `LogisticRegression` e `RandomForest`;
- benchmark opcional com `XGBoost` quando a dependência estiver instalada;
- métricas de validação cruzada;
- classificação por classe, matriz de confusão e curvas ROC/PR;
- threshold otimizado para F1;
- metadata de preprocessing e fingerprint do dataset.

Para habilitar o benchmark opcional com `XGBoost`, instale o extra:

```bash
uv sync --extra benchmark
```

## Automação com `just`

```bash
just install
just lint
just test
just train
just serve-docs
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
