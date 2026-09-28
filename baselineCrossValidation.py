import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

ARQUIVO_TREINO = "train.xlsx"

COLUNA_TEXTO = "resp_text"

COLUNA_CLASSE = "clarity"

RANDOM_STATE = 42


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

df = pd.read_excel(ARQUIVO_TREINO)

# Garante que todos os textos sejam tratados como strings
df[COLUNA_TEXTO] = df[COLUNA_TEXTO].fillna("").astype(str)

print("Dimensões:", df.shape)

print("\nColunas:")
print(df.columns.tolist())

print("\nDistribuição das classes:")
print(df[COLUNA_CLASSE].value_counts())


# ============================================================
# 3. SEPARAÇÃO ENTRE ENTRADA (X) E CLASSE (y)
# ============================================================

X = df[COLUNA_TEXTO]

y = df[COLUNA_CLASSE]


# ============================================================
# 4. VALIDAÇÃO CRUZADA
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 5. PIPELINE: TF-IDF + REGRESSÃO LOGÍSTICA
# ============================================================

pipeline_baseline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("logistic", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])


# ============================================================
# 6. EXECUÇÃO DA VALIDAÇÃO CRUZADA
# ============================================================

scores = cross_val_score(
    pipeline_baseline,
    X,
    y,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)


# ============================================================
# 7. RESULTADOS
# ============================================================

print("\n==============================")

print("VALIDAÇÃO CRUZADA - BASELINE")

print("==============================")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print(f"\nF1 macro médio: {scores.mean():.4f}")

print(f"Desvio padrão: {scores.std():.4f}")


# ============================================================
# 8. REGISTRO DO RESULTADO
# ============================================================

resultado_baseline_cv = {
    "modelo": "TF-IDF + Regressão Logística",
    "folds": 5,
    "f1_macro_medio": scores.mean(),
    "desvio_padrao": scores.std()
}

print("\nResultado salvo:")

print(resultado_baseline_cv)