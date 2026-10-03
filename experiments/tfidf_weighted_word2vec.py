import pandas as pd

from sklearn.model_selection import GridSearchCV

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    RANDOM_STATE,
    TUNING_TFIDF_WEIGHTED_WORD2VEC_FILE,
)

from src.data import carregar_dados_treino

from src.evaluation import (
    avaliar_modelo,
    criar_cv,
)

from src.models import (
    criar_pipeline_tfidf_weighted_word2vec,
)

from src.results import (
    registrar_experimento,
    salvar_tabela,
)


# ============================================================
# DADOS
# ============================================================

X, y = carregar_dados_treino()


# ============================================================
# PIPELINE
# ============================================================

pipeline = criar_pipeline_tfidf_weighted_word2vec()


# ============================================================
# GRID EXPLORATÓRIO
# ============================================================

param_grid = {
    "embedding__vector_size": [
        100,
        200,
    ],
    "embedding__window": [
        5,
    ],
    "embedding__min_count": [
        2,
    ],
    "embedding__epochs": [
        5,
        10,
    ],
    "embedding__sg": [
        1,
    ],
    "logistic__C": [
        0.25,
        0.50,
        1.00,
    ],
}


print("\n==============================")
print("TF-IDF WEIGHTED WORD2VEC")
print("==============================")

print(
    "Configurações:",
    (
        2
        * 1
        * 1
        * 2
        * 1
        * 3
    ),
)

print(
    "Folds de exploração:",
    N_SPLITS_EXPLORATORIA,
)


# ============================================================
# GRID SEARCH
# ============================================================

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=criar_cv(N_SPLITS_EXPLORATORIA),
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    },
    refit="accuracy",
    n_jobs=-1,
    return_train_score=True,
    error_score="raise",
)

grid.fit(X, y)


# ============================================================
# RESULTADOS 3-FOLD
# ============================================================

resultados = pd.DataFrame(
    grid.cv_results_
)

resultados = resultados[
    [
        "param_embedding__vector_size",
        "param_embedding__window",
        "param_embedding__min_count",
        "param_embedding__epochs",
        "param_embedding__sg",
        "param_logistic__C",
        "mean_train_accuracy",
        "std_train_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_train_f1_macro",
        "mean_test_f1_macro",
    ]
].copy()


resultados.rename(
    columns={
        "param_embedding__vector_size":
            "vector_size",
        "param_embedding__window":
            "window",
        "param_embedding__min_count":
            "min_count",
        "param_embedding__epochs":
            "epochs",
        "param_embedding__sg":
            "sg",
        "param_logistic__C":
            "C",
        "mean_train_accuracy":
            "train_accuracy_3fold",
        "std_train_accuracy":
            "train_accuracy_std_3fold",
        "mean_test_accuracy":
            "accuracy_3fold",
        "std_test_accuracy":
            "accuracy_std_3fold",
        "mean_train_f1_macro":
            "train_f1_macro_3fold",
        "mean_test_f1_macro":
            "f1_macro_3fold",
    },
    inplace=True,
)


resultados["accuracy_gap_3fold"] = (
    resultados["train_accuracy_3fold"]
    - resultados["accuracy_3fold"]
)


resultados.sort_values(
    "accuracy_3fold",
    ascending=False,
    inplace=True,
)


resultados.reset_index(
    drop=True,
    inplace=True,
)


print("\n==============================")
print("RESULTADOS - 3-FOLD")
print("==============================")


print(
    resultados.to_string(
        index=False
    )
)


# ============================================================
# CONFIRMAÇÃO 5-FOLD
# ============================================================

_, resumo, resultados_folds = avaliar_modelo(
    grid.best_estimator_,
    X,
    y,
    n_splits=N_SPLITS,
)


melhor_config = {
    "vector_size": int(
        grid.best_params_[
            "embedding__vector_size"
        ]
    ),
    "window": int(
        grid.best_params_[
            "embedding__window"
        ]
    ),
    "min_count": int(
        grid.best_params_[
            "embedding__min_count"
        ]
    ),
    "epochs": int(
        grid.best_params_[
            "embedding__epochs"
        ]
    ),
    "sg": int(
        grid.best_params_[
            "embedding__sg"
        ]
    ),
    "C": float(
        grid.best_params_[
            "logistic__C"
        ]
    ),
}


print("\n==============================")
print("CONFIRMAÇÃO - 5-FOLD")
print("==============================")


print(
    "Melhor configuração:",
    melhor_config,
)


print(
    "\nResultados por fold:"
)

print(
    resultados_folds.to_string(
        index=False
    )
)


print(
    f"\nAccuracy: "
    f"{resumo['accuracy']:.4f} "
    f"± "
    f"{resumo['accuracy_std']:.4f}"
)

print(
    f"F1 Macro: "
    f"{resumo['f1_macro']:.4f} "
    f"± "
    f"{resumo['f1_macro_std']:.4f}"
)

print(
    f"Accuracy treino: "
    f"{resumo['train_accuracy']:.4f}"
)

print(
    f"Gap: "
    f"{resumo['accuracy_gap']:.4f}"
)


# ============================================================
# REGISTRO
# ============================================================

registrar_experimento(
    {
        "experiment_id":
            "tuning_tfidf_weighted_word2vec",

        "experiment_name":
            "TF-IDF-weighted Word2Vec",

        "category":
            "feature_learning",

        "representation":
            "TF-IDF-weighted Word2Vec",

        "model":
            "Regressão Logística",

        "evaluation":
            "3fold_selection_plus_5fold_confirmation",

        "folds":
            N_SPLITS,

        **resumo,

        "selection_accuracy":
            float(grid.best_score_),

        "parameters":
            melhor_config,

        "notes": (
            "Word2Vec treinado dentro de cada fold. "
            "Cada documento é representado por uma média "
            "ponderada dos embeddings pelos pesos TF-IDF."
        ),
    }
)


salvar_tabela(
    resultados,
    TUNING_TFIDF_WEIGHTED_WORD2VEC_FILE,
)


print(
    "\nTabela salva em:",
    TUNING_TFIDF_WEIGHTED_WORD2VEC_FILE,
)