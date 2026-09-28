import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from registro_resultados import registrar_resultado


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

ARQUIVO_TREINO = "train.xlsx"

COLUNA_TEXTO = "resp_text"
COLUNA_CLASSE = "clarity"

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

df = pd.read_excel(ARQUIVO_TREINO)

df[COLUNA_TEXTO] = df[COLUNA_TEXTO].fillna("").astype(str)

print("Dimensões:", df.shape)

print("\nColunas:")
print(df.columns.tolist())

print("\nDistribuição das classes:")
print(df[COLUNA_CLASSE].value_counts())

X = df[COLUNA_TEXTO]
y = df[COLUNA_CLASSE]


# ============================================================
# 3. VALIDAÇÃO CRUZADA
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 4. PIPELINE: TF-IDF + REGRESSÃO LOGÍSTICA
# ============================================================

pipeline_baseline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("logistic", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])


# ============================================================
# 5. VALIDAÇÃO CRUZADA
# ============================================================

# A acurácia é a métrica principal do trabalho.
# O F1 macro é registrado como métrica complementar.

scores = cross_validate(
    pipeline_baseline,
    X,
    y,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    },
    n_jobs=-1,
    return_train_score=True
)


# ============================================================
# 6. RESULTADOS POR FOLD
# ============================================================

resultados_folds = pd.DataFrame({
    "fold": range(1, N_SPLITS + 1),
    "acuracia_treino": scores["train_accuracy"],
    "acuracia_validacao": scores["test_accuracy"],
    "f1_macro_treino": scores["train_f1_macro"],
    "f1_macro_validacao": scores["test_f1_macro"]
})

print("\n==============================")
print("VALIDAÇÃO CRUZADA - BASELINE")
print("==============================")

print(resultados_folds.to_string(index=False))


# ============================================================
# 7. RESUMO
# ============================================================

acuracia_media = float(scores["test_accuracy"].mean())
acuracia_desvio = float(scores["test_accuracy"].std())

f1_macro_medio = float(scores["test_f1_macro"].mean())
f1_macro_desvio = float(scores["test_f1_macro"].std())

acuracia_treino_media = float(scores["train_accuracy"].mean())
f1_treino_medio = float(scores["train_f1_macro"].mean())


print(f"\nAcurácia média:        {acuracia_media:.4f}")
print(f"Desvio padrão:         {acuracia_desvio:.4f}")
print(f"F1 macro médio:        {f1_macro_medio:.4f}")
print(f"Desvio F1 macro:       {f1_macro_desvio:.4f}")
print(f"Acurácia treino média: {acuracia_treino_media:.4f}")
print(f"F1 treino médio:       {f1_treino_medio:.4f}")


# ============================================================
# 8. REGISTRO DO RESULTADO
# ============================================================

resultado_baseline_cv = {
    "experimento": "5-Fold CV - Baseline",
    "representacao": "TF-IDF",
    "modelo": "Regressão Logística",
    "folds": int(N_SPLITS),
    "acuracia_media": acuracia_media,
    "desvio_acuracia": acuracia_desvio,
    "f1_macro_medio": f1_macro_medio,
    "desvio_f1_macro": f1_macro_desvio,
    "acuracia_treino_media": acuracia_treino_media,
    "f1_macro_treino_medio": f1_treino_medio
}

print("\nResultado:")
print(resultado_baseline_cv)

registrar_resultado(resultado_baseline_cv)
