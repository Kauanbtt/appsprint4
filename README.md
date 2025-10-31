# Classificador de Suspeitas (Streamlit)

![Status](https://img.shields.io/badge/status-pronto_para_demo-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Stack](https://img.shields.io/badge/stack-Streamlit%20%7C%20pandas%20%7C%20scikit--learn-orange)

## 1. Visão geral
Aplicativo web (Streamlit) para **classificar anúncios/produtos como suspeitos** com base em um dataset enriquecido. O foco é **velocidade de inicialização** e **simplicidade de uso**: upload de CSV → treino rápido (RandomForest) → métricas e matriz de confusão. Existe também um modo opcional de *tuning* (GridSearch) priorizando **recall**.

> **Importante (deploy rápido):** inclua **`runtime.txt`** com `3.12` para forçar Python 3.12 no Streamlit Cloud. Isso evita builds de fonte no Python 3.13 (que geram erros e lentidão) e garante uso de **wheels pré-compilados** de `numpy/pandas/scikit-learn`.

## 2. Dataset de base
O app usa como padrão o arquivo **`dados_enriquecidos_com_alertas.csv`** (coloque na raiz do projeto) e espera a variável alvo **`alerta_suspeita`**. As colunas mais úteis incluem:
- **Numéricas:** `preco`, `quantidade_vendida`, `avaliacao_nota`, `avaliacao_numero`, `reviews_1_estrelas_pct`, `reviews_5_estrelas_pct`, `rendimento_paginas`, `custo_por_pagina`, `capacidade_num` (derivada de `capacidade` quando existir).
- **Categóricas:** `status_vendedor`, `reputacao_cor`, `categoria_produto`, `modelo_cartucho`.

> O enriquecimento e o uso de `alerta_suspeita` como alvo seguem a base conceitual do projeto. Mantenha o CSV separado por `;`.

## 3. Como executar

### 3.1 Local
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### 3.2 Streamlit Cloud (deploy recomendado)
1. **Arquivos no repositório** (apenas o essencial para builds rápidos):
   - `app.py`
   - `requirements.txt`
   - `runtime.txt`  ← **conteúdo:** `3.12`
   - `README.md`
   - `dados_enriquecidos_com_alertas.csv` *(opcional)*
2. No painel do Streamlit Cloud, aponte o deploy para **`app.py`**.

> **Por que `runtime.txt`?** Seu log mostrou que o ambiente padrão estava em **Python 3.13.9**, e o instalador tentou **compilar** `scikit-learn` + dependências, exigindo `numpy==2.0.0rc1` — pacote não disponível — causando falha e lentidão. Com **3.12**, o pip baixa **wheels prontos** sem compilar.  

## 4. Uso
1. Envie um **CSV separado por `;`** com a coluna alvo `alerta_suspeita` (ou deixe `dados_enriquecidos_com_alertas.csv` na raiz).
2. Clique em **“⚡ Treinar (RÁPIDO)”** para resultados imediatos (RandomForest leve).
3. Opcionalmente, **“🔎 Tuning (GridSearch — recall)”** para buscar hiperparâmetros com melhor *recall*.

## 5. Decisões de design
- **Deploy rápido:** imports do scikit-learn são feitos **no clique**, reduzindo o tempo de boot.
- **Cache seletivo:** `st.cache_data` (CSV) e `st.cache_resource` (preprocessador).
- **Foco em recall:** prioriza **não perder suspeitos**, alinhado à lógica antifraude.
- **Clareza de operação:** UI minimalista; relatório e matriz de confusão visíveis no app.

## 6. Notas dos Speakers (conceituais)
**Ana C. Martins — Produto & Governança**  
> “Reduzimos ruído e criamos um **ritual único de decisão**. O app oferece linguagem comum, previsibilidade e trilho de melhoria contínua. A escolha por *recall* nasce do apetite de risco: **é pior deixar passar um suspeito** do que revisar um falso positivo.”

**Lancelot C. Rodrigues — Arquitetura orientada a valor**  
> “Não perguntamos ‘qual modelo?’, e sim ‘**qual decisão precisa ficar óbvia?**’. A solução remove atrito e facilita consenso. É um **sistema de alinhamento**: padroniza critérios e acelera a leitura de risco sem depender de heróis.”

**Kauan B. — Estratégia & Mudança comportamental**  
> “Mediu-se o custo real dos erros e concentrou-se energia onde há **maior probabilidade de dano**. Três dores foram atacadas: 1) incerteza do que olhar, 2) critérios inconsistentes, 3) falta de narrativa para justificar decisões.”

## 7. Roadmap curto
- Persistir melhor modelo (`.pkl`) e endpoint de inferência.
- Threshold ajustável por custo (otimizar precisão × recall).
- Monitoramento de *drift* e retraining periódico.