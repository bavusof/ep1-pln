from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    N_SPLITS,
    RANDOM_STATE,
)
from src.data import carregar_dados_treino
from src.evaluation import avaliar_modelo
from registro_resultados import registrar_resultado


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

X, y = carregar_dados_treino()

print("Quantidade de exemplos:", len(X))

print("\nDistribuição das classes:")
print(y.value_counts())


# ============================================================
# PIPELINE BASELINE
# ============================================================

pipeline_baseline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(),
    ),
    (
        "logistic",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    ),
])


# ============================================================
# VALIDAÇÃO CRUZADA
# ============================================================

# A função avaliar_modelo centraliza a metodologia de
# validação utilizada pelo projeto.
#
# A mesma função poderá ser reutilizada posteriormente
# para comparar diferentes modelos e representações.
scores, resumo, resultados_folds = avaliar_modelo(
    pipeline_baseline,
    X,
    y,
)


# ============================================================
# RESULTADOS POR FOLD
# ============================================================

print("\n==============================")
print("BASELINE - 5-FOLD CV")
print("==============================")

print(
    resultados_folds.to_string(index=False)
)


# ============================================================
# RESUMO
# ============================================================

print("\n==============================")
print("RESUMO")
print("==============================")

print(
    f"Accuracy: "
    f"{resumo['acuracia_media']:.4f} "
    f"± {resumo['desvio_acuracia']:.4f}"
)

print(
    f"F1 Macro: "
    f"{resumo['f1_macro_medio']:.4f} "
    f"± {resumo['desvio_f1_macro']:.4f}"
)

print(
    f"Accuracy treino: "
    f"{resumo['acuracia_treino_media']:.4f}"
)

print(
    f"F1 Macro treino: "
    f"{resumo['f1_macro_treino_medio']:.4f}"
)


# ============================================================
# REGISTRO DO RESULTADO
# ============================================================

resultado = {
    "experimento": "5-Fold CV - Baseline",
    "representacao": "TF-IDF",
    "modelo": "Regressão Logística",

    "folds": N_SPLITS,

    "acuracia_media":
        resumo["acuracia_media"],

    "desvio_acuracia":
        resumo["desvio_acuracia"],

    "f1_macro_medio":
        resumo["f1_macro_medio"],

    "desvio_f1_macro":
        resumo["desvio_f1_macro"],

    "acuracia_treino_media":
        resumo["acuracia_treino_media"],

    "f1_macro_treino_medio":
        resumo["f1_macro_treino_medio"],
}

registrar_resultado(resultado)