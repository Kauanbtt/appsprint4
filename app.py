# -*- coding: utf-8 -*-
"""
Classificador de Suspeitas — Streamlit (boot rápido)
- Sem installs em runtime (usar requirements.txt)
- Imports tardios do scikit-learn (só quando necessário)
- Cache para dataset e preprocessor
- Modo RÁPIDO por padrão (apenas RandomForest leve); Tuning opcional
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Classificador de Suspeitas", page_icon="🧪", layout="wide")
st.title("🧪 Classificador de Suspeitas — Versão Rápida")

st.markdown(
    "Envie um CSV **`;`-separado** com a coluna alvo `alerta_suspeita`. "
    "Se não enviar, o app usa `dados_enriquecidos_com_alertas.csv` da pasta do projeto."
)

# -------------------- Cache & Helpers --------------------
@st.cache_data(show_spinner=False)
def load_dataframe(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep=';')
    default = 'dados_enriquecidos_com_alertas.csv'
    return pd.read_csv(default, sep=';') if os.path.exists(default) else None

def ensure_capacidade_num(df: pd.DataFrame) -> pd.DataFrame:
    if 'capacidade' in df.columns and 'capacidade_num' not in df.columns:
        df['capacidade_num'] = (
            df['capacidade'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        )
    elif 'capacidade_num' not in df.columns:
        df['capacidade_num'] = np.nan
    return df

@st.cache_resource(show_spinner=False)
def build_preprocessor(num_cols, cat_cols):
    # Imports tardios
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # OneHot compatível c/ versões
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre_num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    pre_cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")), ("ohe", ohe)])
    return ColumnTransformer([("num", pre_num, num_cols), ("cat", pre_cat, cat_cols)], remainder="passthrough")

def quick_train(Xtr, Xte, ytr, yte, preprocessor):
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=None, min_samples_leaf=1,
        random_state=42, class_weight="balanced"
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return {
        "acc": float(accuracy_score(yte, pred)),
        "report": classification_report(yte, pred, target_names=["Não Suspeito (0)", "Suspeito (1)"]),
        "cm": confusion_matrix(yte, pred).tolist(),
    }

def grid_tune(Xtr, Xte, ytr, yte, preprocessor):
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    pipe = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))])
    grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [10, 20, None],
        "clf__min_samples_leaf": [1, 3],
    }
    gs = GridSearchCV(pipe, grid, cv=3, n_jobs=-1, scoring="recall", verbose=0)
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    pred = best.predict(Xte)
    return {
        "best_params": gs.best_params_,
        "best_cv_recall": float(gs.best_score_),
        "acc": float(accuracy_score(yte, pred)),
        "report": classification_report(yte, pred, target_names=["Não Suspeito (0)", "Suspeito (1)"]),
        "cm": confusion_matrix(yte, pred).tolist(),
    }

# -------------------- Data --------------------
uploaded = st.file_uploader("📂 Envie seu CSV", type=["csv"])
df = load_dataframe(uploaded)

if df is None:
    st.error("Nenhum CSV enviado e `dados_enriquecidos_com_alertas.csv` não encontrado.")
    st.stop()

df = ensure_capacidade_num(df)

if "alerta_suspeita" not in df.columns:
    st.error("Coluna alvo `alerta_suspeita` não encontrada no CSV.")
    st.stop()

with st.expander("ℹ️ Colunas esperadas (usadas se existirem)", expanded=False):
    st.write({
        "numéricas": [
            "preco","quantidade_vendida","avaliacao_nota","avaliacao_numero",
            "reviews_1_estrelas_pct","reviews_5_estrelas_pct","rendimento_paginas",
            "custo_por_pagina","capacidade_num"
        ],
        "categóricas": ["status_vendedor","reputacao_cor","categoria_produto","modelo_cartucho"],
        "alvo": "alerta_suspeita"
    })

st.write("### Distribuição do alvo")
st.dataframe(df["alerta_suspeita"].value_counts().to_frame("contagem"))

NUM_COLS = [c for c in [
    "preco","quantidade_vendida","avaliacao_nota","avaliacao_numero",
    "reviews_1_estrelas_pct","reviews_5_estrelas_pct","rendimento_paginas",
    "custo_por_pagina","capacidade_num"
] if c in df.columns]
CAT_COLS = [c for c in ["status_vendedor","reputacao_cor","categoria_produto","modelo_cartucho"] if c in df.columns]

X = df[NUM_COLS + CAT_COLS]
y = df["alerta_suspeita"]

from sklearn.model_selection import train_test_split
strat = y if y.nunique() > 1 else None
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)

pre = build_preprocessor(NUM_COLS, CAT_COLS)

col1, col2 = st.columns(2)

with col1:
    if st.button("⚡ Treinar (RÁPIDO)"):
        with st.spinner("Treinando RandomForest leve..."):
            res = quick_train(Xtr, Xte, ytr, yte, pre)
        st.subheader("Resultado — Modo Rápido (RF)")
        st.metric("Acurácia (teste)", f"{res['acc']:.4f}")
        st.text(res["report"])
        st.dataframe(pd.DataFrame(res["cm"], index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]))
        st.caption("Dica: utilize este modo para validação rápida e apresentações.")

with col2:
    if st.button("🔎 Tuning (GridSearch — recall)"):
        with st.spinner("Executando GridSearchCV..."):
            tuned = grid_tune(Xtr, Xte, ytr, yte, pre)
        st.subheader("RandomForest — Melhor Configuração (Recall)")
        st.write("**Parâmetros:**", tuned["best_params"])
        st.write("**Recall (CV):**", f"{tuned['best_cv_recall']:.4f}")
        st.metric("Acurácia (teste)", f"{tuned['acc']:.4f}")
        st.text(tuned["report"])
        st.dataframe(pd.DataFrame(tuned["cm"], index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]))

st.caption("© 2025 — App otimizado para deploy rápido (imports tardios + cache + modo rápido).")