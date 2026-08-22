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

Para ativar o ambiente virtual local no PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Em shells Unix:

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
A API de inferência foi implementada com FastAPI em `src/api/api.py`, expondo `/health`
e `/predict`, com validação de payload, erros padronizados e logs por requisição.

Para executar a API localmente:

```bash
just api
```

Para executar uma checagem de drift entre dois CSVs:

```bash
just drift reference=data/reference.csv current=data/current.csv
```

O resultado é salvo em `models/reports/drift_report.json` e os eventos operacionais em
`logs/operational_metrics.jsonl`.

O treino passa a registrar no MLflow:

- benchmark com `LogisticRegression` e `RandomForest`;
- benchmark opcional com `XGBoost` quando a dependência estiver instalada;
- métricas de validação cruzada;
- classificação por classe, matriz de confusão e curvas ROC/PR;
- threshold otimizado para F1;
- metadata de preprocessing e fingerprint do dataset.
- versão do modelo no MLflow Model Registry, com status de governança.

Para habilitar o benchmark opcional com `XGBoost`, instale o extra:

```bash
uv sync --extra benchmark
```

## MLflow Model Registry

Cada execução de treino registra o artefato `runs:/<run_id>/model` no Model Registry quando
`MLFLOW_ENABLE_MODEL_REGISTRY=true`. O nome do registered model é derivado do experimento
(`online-shoppers-purchasing-intention-random-forest`) ou definido por
`MLFLOW_REGISTERED_MODEL_NAME`.

O projeto usa os status `Staging`, `Production` e `Archived` como stages e aliases do MLflow,
além das tags `governance_status`, `approval_status`, `approved_by` e motivos de promoção ou
rollback. O treino cria versões inicialmente em `Staging` com aprovação `pending`; promoções para
`Production` exigem aprovador e aprovação explícita.

```bash
uv run python -m src.ml_project.model_registry promote \
  --version 1 \
  --target-status Production \
  --approver "nome.aprovador" \
  --reason "Metricas aprovadas para producao"
```

```bash
uv run python -m src.ml_project.model_registry rollback \
  --version 1 \
  --approver "nome.aprovador" \
  --reason "Regressao detectada na versao atual"
```

A trilha de governança fica registrada em `models/model_registry.json`,
`models/model_registry_events.json` e também como artefatos do run em `registry/`. Para carregar
a versão aprovada na inferência, configure `MLFLOW_USE_MODEL_REGISTRY_FOR_INFERENCE=true` e, se
necessário, ajuste `MLFLOW_INFERENCE_MODEL_ALIAS=Production`.

## Automação com `just`

```bash
just install
just lint
just test
just train
just serve-docs
```

## Variáveis de ambiente

Copie o arquivo `.env.example` para `.env` e ajuste os valores. Em produção, o serviço de
inferência usa o alias `Production` do Model Registry no Docker Compose; fora do container,
o fallback local deve ser habilitado explicitamente com
`MLFLOW_USE_MODEL_REGISTRY_FOR_INFERENCE=false`.

## Docker

O `Dockerfile` utiliza multi-stage build, separando a etapa de build da etapa de runtime.

Para validar as imagens e executar um smoke test:

```bash
just docker-build
just docker-smoke
```

O smoke test desabilita explicitamente o Model Registry porque o backend filesystem do MLflow
não suporta criação de versões em alguns volumes bind-mounted com usuário não-root. Para
governança em container, use um backend MLflow/DB compartilhado e habilite
`MLFLOW_ENABLE_MODEL_REGISTRY=true`.

## Contribuição

1. Crie uma branch para sua feature.
2. Execute lint e testes antes do commit.
3. Mantenha o pipeline e o tracking de experimentos reprodutíveis.

## Licença

MIT
