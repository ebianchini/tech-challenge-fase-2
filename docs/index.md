# Tech Challenge

## Grupo MLEKs

## Documentação do projeto final da Fase 2 (MLET10).

* * *

Esta documentação centraliza o contexto do projeto (ML Canvas e dataset), a estrutura do repositório, e como executar o ambiente localmente. Para entender o projeto, comece pelo [ML Canvas](mlcanvas/) e pelo [Dataset](dataset/).

[Primeiros passos](getting-started/)<br />
[Comandos](comandos/)<br />
[Arquitetura](arquitetura/)<br />
[ML Canvas](mlcanvas/)<br />
[Dataset](dataset/)<br />
[Comparativo entre Modelos](comparativo-modelos/)<br />

## Recursos

### Setup em um comando

O ambiente e as dependências são gerenciados com `uv`. Use `just install` para criar a virtualenv e sincronizar o projeto.

### Qualidade e testes

Padronize o codigo com `ruff` e rode testes com `pytest`. Targets: `make lint`, `make format`, `make test`.

### Documentacao versionada

As paginas ficam em `docs/docs/` e o site gerado em `docs/site/`. Use `make docs` para visualizar localmente.

### Estrutura do projeto

A pagina [Estrutura do projeto](estrutura-do-projeto/) descreve as pastas do repositorio e onde colocar dados, notebooks e artefatos.

### Arquitetura versionada

A pagina [Arquitetura](arquitetura/) mostra o pipeline de ML e o fluxo da API com diagramas Mermaid versionados em Markdown.

### ML Canvas

A pagina [ML Canvas](mlcanvas/) consolida a definicao do problema, metricas de sucesso, dados e riscos do projeto.

### Dataset

A pagina [Dataset](dataset/) descreve a fonte e o dicionario de dados do Telco Customer Churn (IBM).

### Comparativo entre Modelos

A pagina [Comparativo entre Modelos](comparativo-modelos/) demonstra uma comparação de métricas e custo/benefício entre alguns modelos testados.