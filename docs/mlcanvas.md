# ML Canvas: Online Shoppers Purchasing Intention

## 1. Proposta de valor

#### Problema de negócio

Baixa taxa de conversão no site de e-commerce e perda de oportunidades de vendas em tempo real.

#### Solução de Machine Learning

Um modelo preditivo capaz de identificar, durante a sessão de navegação, se o usuário tem propensão a realizar uma transação, com base no dataset Online Shoppers Purchasing Intention.

## 2. Decisões e ações

#### Quem usa?

Sistema automatizado de marketing e equipe de experiência do usuário (UX/UI).

#### Qual decisão ou ação é tomada?

- Acionar gatilhos de retenção em tempo real, como chat de suporte proativo, cupom de desconto dinâmico ou oferta personalizada, para usuários com alta propensão de compra que demonstrem sinais de saída.
- Otimizar campanhas direcionadas para visitantes novos e recorrentes.

## 3. Métricas

#### Métrica de negócio

- Aumento da taxa de conversão global.
- Redução do abandono de carrinho e de sessão.

#### Métrica de Machine Learning

- F1-Score e ROC-AUC, por serem essenciais diante do desbalanceamento natural de classes em e-commerce.
- Precision, para evitar custos desnecessários ao oferecer descontos para quem já compraria organicamente.

## 4. Fontes de dados

#### Origem

Arquivo CSV do Kaggle, referente ao dataset Online Shoppers Purchasing Intention.

#### Volume

12.330 sessões de usuários coletadas ao longo de um ano.

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

- Desbalanceamento severo de classes, com muitas sessões sem compra e poucas com compra.
- Mudanças sazonais drásticas que o histórico de um ano pode não cobrir totalmente.

#### Restrições

O modelo precisa ser leve o suficiente para permitir inferências rápidas e baixa latência, caso seja integrado em tempo real no site.
