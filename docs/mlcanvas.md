# ML Canvas: Online Shoppers Purchasing Intention

## 1. Proposta de valor

#### Problema de negócio

Baixa taxa de conversão no site de e-commerce e perda de oportunidades de vendas em tempo real.

#### Solução de Machine Learning

Um classificador Random Forest capaz de estimar, durante a sessão de navegação, a propensão de o usuário realizar uma transação, com base no dataset Online Shoppers Purchasing Intention. A decisão de ativar uma ação deve usar um limiar calibrado em validação, e não apenas a classe padrão do modelo.

## 2. Decisões e ações

#### Quem usa?

Sistema automatizado de marketing e equipe de experiência do usuário (UX/UI).

#### Qual decisão ou ação é tomada?

- Priorizar gatilhos de retenção em tempo real, como chat de suporte proativo, cupom de desconto dinâmico ou oferta personalizada, para sessões classificadas como propensas à conversão e que demonstrem sinais de saída.
- Otimizar campanhas direcionadas para visitantes novos e recorrentes.

O limiar ajustado observado foi `0,510996`, com F1 de `0,680574`. Como esse valor é próximo de 0,5, o ajuste atual é uma pequena mudança no ponto de corte, não uma mudança estrutural da política de decisão. Em produção, o limiar deve ser escolhido conforme o custo de falsos positivos, falsos negativos e descontos concedidos.

## 3. Métricas

#### Métrica de negócio

- Aumento da taxa de conversão global.
- Redução do abandono de carrinho e de sessão.

#### Métrica de Machine Learning

- F1-Score e PR-AUC como métricas principais para a classe `Revenue=True`, porque a taxa da classe minoritária é `15,63%` e a precisão-recall representa melhor o desempenho sobre os compradores do que a acurácia isolada.
- ROC-AUC como métrica complementar de capacidade de ordenação; o resultado de `0,9266` no conjunto avaliado e `0,9222` na validação cruzada indica boa separação entre sessões com e sem compra.
- Precision (`0,6650`) para controlar o custo de abordar sessões que não converteriam e recall (`0,6911`) para limitar a perda de oportunidades de compra.

#### Resultado do modelo base

No conjunto de avaliação, o Random Forest apresentou:

| Indicador | Resultado |
| --- | ---: |
| Acurácia | 0,8972 |
| Precision | 0,6650 |
| Recall | 0,6911 |
| F1-Score | 0,6778 |
| PR-AUC | 0,7065 |
| ROC-AUC | 0,9266 |

Na validação cruzada, os valores médios foram F1 `0,6844`, PR-AUC `0,6950` e ROC-AUC `0,9222`. A proximidade entre a validação cruzada e a avaliação final reduz a indicação de uma degradação evidente, mas não substitui validação temporal e monitoramento após a entrada em produção.

O threshold tuning elevou a acurácia de `0,8972` para `0,8996` e o F1 de `0,6778` para `0,6806`; a precision subiu de `0,6650` para `0,6779`. O ganho é pequeno, portanto a escolha do limiar deve ser justificada por uma função de custo e por experimento online, não pelo F1 isoladamente. A regressão logística teve F1 médio (`0,6276`) e PR-AUC médio (`0,6499`) inferiores aos do Random Forest, embora tenha apresentado ROC-AUC médio (`0,9030`) também competitivo.

## 4. Fontes de dados

#### Origem

Arquivo CSV do Kaggle, referente ao dataset Online Shoppers Purchasing Intention.

#### Volume

12.330 sessões de usuários coletadas ao longo de um ano. O treinamento utilizou `16.476` linhas após reamostragem; esse número não representa novas sessões observadas e deve ser distinguido do volume original ao comunicar cobertura e generalização.

#### Atributos principais

- Comportamentais de navegação: Administrative, Administrative Duration, Informational, Informational Duration, Product Related, Product Related Duration.
- Métricas do Google Analytics: Bounce Rate, Exit Rate, Page Value.
- Contextuais e temporais: Special Day, Month, Weekend.
- Demográficos e técnicos: OperatingSystems, Browser, Region, TrafficType, VisitorType.

#### Variável alvo

Revenue, uma variável booleana que indica se a sessão finalizou em compra.

## 5. Premissas, riscos e restrições

#### Premissas

O comportamento passado extraído das sessões reflete padrões que se mantêm úteis para previsões futuras em tempo de execução.

#### Riscos

- Desbalanceamento relevante de classes: apenas `15,63%` das sessões pertencem à classe positiva. A reamostragem pode melhorar o aprendizado da classe minoritária, mas pode alterar a calibração das probabilidades e não elimina falsos positivos.
- Mudanças sazonais drásticas que o histórico de um ano pode não cobrir totalmente.
- Possível vazamento de informação ou mudança de distribuição em atributos calculados durante a sessão; a disponibilidade de cada variável precisa ser garantida antes da ação em tempo real.
- Acurácia elevada pode mascarar desempenho insuficiente para compradores; decisões de negócio não devem ser aprovadas com base apenas nessa métrica.

#### Restrições

O modelo precisa ser leve o suficiente para permitir inferências rápidas e baixa latência, caso seja integrado em tempo real no site. A política de ação deve preservar um modo sem incentivo financeiro e registrar o limiar aplicado, a versão do modelo e o resultado posterior da sessão.

#### Critério de sucesso em produção

- Monitorar, por versão e segmento, PR-AUC, precision, recall, F1, taxa de conversão e custo por intervenção.
- Avaliar o limiar com dados de conversão posteriores, comparando a política com um grupo de controle.
- Reavaliar o modelo quando houver queda sustentada de PR-AUC, mudança na taxa de compradores ou aumento do custo por conversão.
