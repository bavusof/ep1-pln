import pandas as pd

from sklearn.model_selection import GridSearchCV

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    RANDOM_STATE,
    TUNING_RESULTS_DIR,
)

from src.data import carregar_dados_treino

from src.evaluation import (
    avaliar_modelo,
    criar_cv,
)

from src.models import (
    criar_pipeline_tfidf_word_structural,
)

from src.results import (
    registrar_experimento,
    salvar_tabela,
)


# ============================================================
# CONFIGURAÇÃO BASE
# ============================================================

BASE_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 1.0,
    "sublinear_tf": True,
    "C": 0.5,
}


# ============================================================
# DADOS
# ============================================================

X, y = carregar_dados_treino()


# ============================================================
# PIPELINE
# ============================================================

pipeline = (
    criar_pipeline_tfidf_word_structural(
        **BASE_PARAMS,
    )
)


# ============================================================
# GRID
# ============================================================

param_grid = {
    "features__transformer_weights": [
        {
            "structural": 0.25
        },
        {
            "structural": 0.50
        },
        {
            "structural": 1.00
        },
        {
            "structural": 2.00
        },
    ]
}


# ============================================================
# GRID SEARCH EXPLORATÓRIO
# ============================================================

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=criar_cv(
        N_SPLITS_EXPLORATORIA
    ),
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    },
    refit="accuracy",
    n_jobs=2,
    return_train_score=True,
    error_score="raise",
)


print("\n==============================")
print("FEATURE ENGINEERING")
print("==============================")

print(
    "Configuração base:",
    BASE_PARAMS,
)

print(
    "Pesos estruturais:",
    [0.25, 0.50, 1.00, 2.00],
)

grid.fit(X, y)


# ============================================================
# RESULTADOS
# ============================================================

resultados = pd.DataFrame(
    grid.cv_results_
)


resultados = resultados[
    [
        "param_features__transformer_weights",
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
        "param_features__transformer_weights":
            "peso_estrutural",

        "mean_train_accuracy":
            "train_accuracy",

        "std_train_accuracy":
            "train_accuracy_std",

        "mean_test_accuracy":
            "accuracy",

        "std_test_accuracy":
            "accuracy_std",

        "mean_train_f1_macro":
            "train_f1_macro",

        "mean_test_f1_macro":
            "f1_macro",
    },
    inplace=True,
)


resultados[
    "peso_estrutural"
] = resultados[
    "peso_estrutural"
].apply(
    lambda x: x["structural"]
)


resultados["accuracy_gap"] = (
    resultados["train_accuracy"]
    - resultados["accuracy"]
)


resultados.sort_values(
    "accuracy",
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

_, resumo, _ = avaliar_modelo(
    grid.best_estimator_,
    X,
    y,
    n_splits=N_SPLITS,
)


print("\n==============================")
print("CONFIRMAÇÃO - 5-FOLD")
print("==============================")


print(
    "Melhor peso estrutural:",
    grid.best_params_[
        "features__transformer_weights"
    ],
)


print(
    f"Accuracy: "
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

resultado_registro = {
    "experiment_id":
        "feature_engineering_structural",

    "experiment_name":
        "TF-IDF + Características Estruturais",

    "category":
        "feature_engineering",

    "representation":
        (
            "TF-IDF palavra + "
            "características estruturais"
        ),

    "model":
        "Regressão Logística",

    "evaluation":
        (
            "3fold_selection_plus_"
            "5fold_confirmation"
        ),

    "folds":
        N_SPLITS,

    **resumo,

    "selection_accuracy":
        float(
            grid.best_score_
        ),

    "parameters":
        {
            **BASE_PARAMS,

            "structural_weight":
                grid.best_params_[
                    "features__"
                    "transformer_weights"
                ],
        },

    "notes":
        (
            "Características estruturais extraídas "
            "do texto e combinadas com TF-IDF. "
            "Pesos testados em exploração de 3 folds "
            f"e confirmação em {N_SPLITS} folds."
        ),
}


registrar_experimento(
    resultado_registro
)


# ============================================================
# SALVAMENTO
# ============================================================

arquivo = (
    TUNING_RESULTS_DIR
    / "feature_engineering.csv"
)


salvar_tabela(
    resultados,
    arquivo,
)


print(
    "\nTabela salva em:",
    arquivo,
)