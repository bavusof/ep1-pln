import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    N_SPLITS,
    RANDOM_STATE,
)
from src.data import carregar_dados_treino
from src.evaluation import criar_cv
from src.results import registrar_experimento


# ============================================================
# CARREGAMENTO
# ============================================================

X, y = carregar_dados_treino()

print("Quantidade de exemplos:", len(X))
print("\nDistribuição das classes:")
print(y.value_counts())


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
        ),
    ),
    (
        "svc",
        LinearSVC(
            random_state=RANDOM_STATE,
        ),
    ),
])


# ============================================================
# PARÂMETROS
# ============================================================

param_grid = {
    "svc__C": [
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ],
}


# ============================================================
# GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=criar_cv(),
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    },
    refit="accuracy",
    n_jobs=-1,
    return_train_score=True,
    error_score="raise",
)


print("\n==============================")
print("LINEAR SVC - GRID SEARCH")
print("==============================")


grid_search.fit(X, y)


# ============================================================
# RESULTADOS
# ============================================================

resultados = pd.DataFrame(
    grid_search.cv_results_
)

resultados = resultados[
    [
        "param_svc__C",
        "mean_train_accuracy",
        "std_train_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_train_f1_macro",
        "mean_test_f1_macro",
    ]
].copy()


resultados = resultados.rename(
    columns={
        "param_svc__C": "C",
        "mean_train_accuracy":
            "accuracy_treino_media",
        "std_train_accuracy":
            "desvio_accuracy_treino",
        "mean_test_accuracy":
            "accuracy_validacao_media",
        "std_test_accuracy":
            "desvio_accuracy_validacao",
        "mean_train_f1_macro":
            "f1_treino_medio",
        "mean_test_f1_macro":
            "f1_validacao_medio",
    }
)


# ============================================================
# GAP
# ============================================================

resultados["gap_accuracy"] = (
    resultados["accuracy_treino_media"]
    - resultados["accuracy_validacao_media"]
)


resultados = (
    resultados
    .sort_values(
        "accuracy_validacao_media",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# EXIBIÇÃO
# ============================================================

print("\n==============================")
print("RESULTADOS")
print("==============================")


print(
    resultados.to_string(
        index=False
    )
)


print("\n==============================")
print("MELHOR CONFIGURAÇÃO")
print("==============================")


print(
    "C:",
    grid_search.best_params_["svc__C"],
)

print(
    f"Accuracy média: "
    f"{grid_search.best_score_:.4f}"
)


melhor_linha = resultados.iloc[0]

print(
    f"F1 Macro médio: "
    f"{melhor_linha['f1_validacao_medio']:.4f}"
)

print(
    f"Gap treino-validação: "
    f"{melhor_linha['gap_accuracy']:.4f}"
)


# ============================================================
# REGISTRO
# ============================================================

resultado = {
    "experiment_id":
        "linear-svc-grid",

    "experiment_name":
        "TF-IDF + LinearSVC",

    "category":
        "tradicional",

    "representation":
        "TF-IDF palavra",

    "model":
        "LinearSVC",

    "evaluation":
        "5-fold CV",

    "folds":
        N_SPLITS,

    "accuracy":
        float(grid_search.best_score_),

    "f1_macro":
        float(
            melhor_linha[
                "f1_validacao_medio"
            ]
        ),

    "accuracy_train_mean":
        float(
            melhor_linha[
                "accuracy_treino_media"
            ]
        ),

    "gap_accuracy":
        float(
            melhor_linha[
                "gap_accuracy"
            ]
        ),

    "parameters":
        str(grid_search.best_params_),
}


registrar_experimento(resultado)


# ============================================================
# SALVAMENTO
# ============================================================

resultados.to_csv(
    "results/linear_svc_grid.csv",
    index=False,
)

print(
    "\nTabela salva em:"
    " results/linear_svc_grid.csv"
)