# Contrato de Inferencia

## 1. Escopo e versao

Este documento define o contrato de entrada e saida para a predicao de conversao de uma sessao do dataset **Online Shoppers Purchasing Intention**.

- Versao do contrato: `1.0`
- Tipo de operacao: classificacao binaria
- Variavel prevista: `Revenue`
- Unidade de predicao: uma sessao de navegacao
- Entrada aceita pelo modulo atual: `pandas.DataFrame`
- Entrada prevista para API: objeto JSON com uma ou mais sessoes

O payload de inferencia nao deve conter `Revenue` nem as features derivadas. O modulo `predict.py` calcula internamente as features de sessao, reaplica o encoding persistido e valida a compatibilidade com o modelo salvo.

## 2. Schema de entrada

Cada item de `instances` deve conter exatamente os campos abaixo. Campos adicionais sao ignorados pelo modulo Python atual, mas nao fazem parte do contrato da API e devem ser rejeitados na camada HTTP.

| Campo | Tipo JSON | Obrigatorio | Restricoes e significado |
| --- | --- | --- | --- |
| `Administrative` | inteiro | sim | Numero de paginas administrativas visitadas; `>= 0` |
| `Administrative_Duration` | numero | sim | Tempo em paginas administrativas; `>= 0` |
| `Informational` | inteiro | sim | Numero de paginas informativas visitadas; `>= 0` |
| `Informational_Duration` | numero | sim | Tempo em paginas informativas; `>= 0` |
| `ProductRelated` | inteiro | sim | Numero de paginas de produto visitadas; `>= 0` |
| `ProductRelated_Duration` | numero | sim | Tempo em paginas de produto; `>= 0` |
| `BounceRates` | numero | sim | Taxa de rejeicao; intervalo esperado `[0, 1]` |
| `ExitRates` | numero | sim | Taxa de saida; intervalo esperado `[0, 1]` |
| `PageValues` | numero | sim | Valor monetario atribuido as paginas; `>= 0` |
| `SpecialDay` | numero | sim | Proximidade de data especial; intervalo esperado `[0, 1]` |
| `Month` | string | sim | Mes: `Feb`, `Mar`, `May`, `June`, `Jul`, `Aug`, `Sep`, `Oct`, `Nov` ou `Dec` |
| `OperatingSystems` | inteiro | sim | Identificador do sistema operacional; `>= 1` |
| `Browser` | inteiro | sim | Identificador do navegador; `>= 1` |
| `Region` | inteiro | sim | Identificador da regiao; `>= 1` |
| `TrafficType` | inteiro | sim | Identificador da origem de trafego; `>= 1` |
| `VisitorType` | string | sim | `Returning_Visitor`, `New_Visitor` ou `Other` |
| `Weekend` | booleano | sim | Indica se a sessao ocorreu no fim de semana |

Regras gerais:

- todos os campos sao obrigatorios;
- valores nulos, `NaN` e infinitos nao sao aceitos;
- contagens e duracoes nao podem ser negativas;
- `Revenue` e as colunas derivadas (`TotalSessionTime`, `TotalPagesVisited` e os seis ratios) sao campos internos e nao devem ser enviados;
- categorias desconhecidas podem ser recusadas pela API. O encoder interno usa `handle_unknown="ignore"`, mas aceitar categorias novas sem governanca pode mascarar erro de integracao;
- o lote deve conter ao menos uma instancia e deve respeitar um limite definido pela API quando ela for implementada.

## 3. Payload JSON

O contrato de transporte recomendado para a API e um objeto com `contract_version` e `instances`:

```json
{
  "contract_version": "1.0",
  "instances": [
    {
      "Administrative": 0,
      "Administrative_Duration": 0.0,
      "Informational": 0,
      "Informational_Duration": 0.0,
      "ProductRelated": 1,
      "ProductRelated_Duration": 0.0,
      "BounceRates": 0.2,
      "ExitRates": 0.2,
      "PageValues": 0.0,
      "SpecialDay": 0.0,
      "Month": "Feb",
      "OperatingSystems": 1,
      "Browser": 1,
      "Region": 1,
      "TrafficType": 1,
      "VisitorType": "Returning_Visitor",
      "Weekend": false
    }
  ]
}
```

Para uso direto do modulo Python, o equivalente e um `DataFrame` com as 17 colunas de entrada, sem a coluna `Revenue`:

```python
import pandas as pd
from src.ml_project.modeling.predict import predict

frame = pd.DataFrame(payload["instances"])
predictions = predict(dataframe=frame)
```

## 4. Schema de saida

A resposta deve preservar a ordem das instancias recebidas. Cada predicao possui:

| Campo | Tipo | Descricao |
| --- | --- | --- |
| `prediction_id` | string | Identificador unico da predicao, gerado pela API |
| `predicted_revenue` | inteiro | Classe prevista: `0` para nao conversao ou `1` para conversao |
| `model_version` | string | Versao ou identificador do artefato usado |
| `contract_version` | string | Versao deste contrato |

Resposta JSON recomendada:

```json
{
  "contract_version": "1.0",
  "model_version": "model.joblib",
  "predictions": [
    {
      "prediction_id": "0",
      "predicted_revenue": 0
    }
  ]
}
```

O modulo Python atual retorna `pd.Series` nomeada `predicted_revenue`, com o mesmo indice e quantidade de linhas da entrada. A camada de API deve converter essa serie para a estrutura JSON acima, sem alterar a ordem das predicoes.

## 5. Erros de contrato

A API deve responder com erro estruturado, sem expor stack trace ou caminhos locais:

```json
{
  "error": {
    "code": "INVALID_INPUT_SCHEMA",
    "message": "Campo obrigatorio ausente: Month",
    "details": ["Month"]
  }
}
```

Codigos reservados:

- `INVALID_INPUT_SCHEMA` para campos ausentes, extras, tipos invalidos ou valores fora das restricoes;
- `MODEL_NOT_FOUND` quando o artefato nao estiver disponivel;
- `MODEL_SCHEMA_MISMATCH` quando o modelo e os metadados forem incompativeis;
- `PREDICTION_FAILED` para falhas inesperadas durante a inferencia.

O modulo atual representa falhas de schema com `ValueError` e falhas de artefato com `FileNotFoundError`; o adaptador HTTP deve mapea-las para os codigos acima.
