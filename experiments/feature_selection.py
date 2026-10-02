import pandas as pd

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    RANDOM_STATE,
    TUNING_FEATURE_SELECTION_FILE,
)

from src.data import carregar_dados_treino

from src.evaluation import (
    criar_grid_search,
    avaliar_modelo,
)

from src.models import (
    criar_pipeline_tfidf_word_selecao,
)

from src.results import (
    registrar_experimento,
    salvar_tabela,
)


# ============================================================
# BASE EXPERIMENTAL
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

print(
    "Quantidade de exemplos:",
    len(X),
)

print(
    "\nDistribuição das classes:"
)

print(
    y.value_counts()
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = criar_pipeline_tfidf_word_selecao(
    **BASE_PARAMS
)


# ============================================================
# GRID
# ============================================================

param_grid = {
    "selection__percentile": [
        10,
        25,
        50,
        75,
        100,
    ],
}


# ============================================================
# EXPLORAÇÃO
# ============================================================

grid = criar_grid_search(
    pipeline,
    param_grid,
    cv=None,
    n_splits=N_SPLITS_EXPLORATORIA,
    n_jobs=2,
)


print("\n==============================")
print("SELEÇÃO DE ATRIBUTOS")
print("==============================")

print(
    "Percentuais testados:",
    param_grid["selection__percentile"],
)

print(
    f"Folds de seleção: "
    f"{N_SPLITS_EXPLORATORIA}"
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
        "param_selection__percentile",
        "mean_train_accuracy",
        "std_train_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
    ]
].copy()


resultados.rename(
    columns={
        "param_selection__percentile":
            "percentile",
        "mean_train_accuracy":
            "train_accuracy_3fold",
        "std_train_accuracy":
            "train_accuracy_std_3fold",
        "mean_test_accuracy":
            "accuracy_3fold",
        "std_test_accuracy":
            "accuracy_std_3fold",
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


print(
    "\nResultados:"
)

print(
    resultados.to_string(
        index=False
    )
)


# ============================================================
# CONFIRMAÇÃO 5-FOLD
# ============================================================

_, confirmacao, _ = avaliar_modelo(
    grid.best_estimator_,
    X,
    y,
    n_splits=N_SPLITS,
)


print("\n==============================")
print("CONFIRMAÇÃO - 5-FOLD")
print("==============================")


print(
    "Melhor percentile:",
    grid.best_params_[
        "selection__percentile"
    ],
)


print(
    f"Accuracy: "
    f"{confirmacao['accuracy']:.4f} "
    f"± "
    f"{confirmacao['accuracy_std']:.4f}"
)


print(
    f"F1 Macro: "
    f"{confirmacao['f1_macro']:.4f} "
    f"± "
    f"{confirmacao['f1_macro_std']:.4f}"
)


print(
    f"Accuracy treino: "
    f"{confirmacao['train_accuracy']:.4f}"
)


print(
    f"Gap: "
    f"{confirmacao['accuracy_gap']:.4f}"
)


# ============================================================
# REGISTRO
# ============================================================

resultado = {
    "experiment_id":
        "feature_selection_tfidf",

    "experiment_name":
        "TF-IDF + Seleção de Atributos",

    "category":
        "feature_selection",

    "representation":
        "TF-IDF palavra + chi2",

    "model":
        "Regressão Logística",

    "evaluation":
        "3fold_selection_plus_5fold_confirmation",

    "folds":
        N_SPLITS,

    **confirmacao,

    "selection_accuracy":
        float(grid.best_score_),

    "parameters":
        {
            **BASE_PARAMS,
            "selection_percentile":
                grid.best_params_[
                    "selection__percentile"
                ],
        },

    "notes":
        (
            "Seleção por qui-quadrado após TF-IDF. "
            f"Exploração com {N_SPLITS_EXPLORATORIA}-fold "
            f"e confirmação com {N_SPLITS}-fold."
        ),
}


registrar_experimento(
    resultado
)


salvar_tabela(
    resultados,
    TUNING_FEATURE_SELECTION_FILE,
)


print(
    "\nTabela salva em:",
    TUNING_FEATURE_SELECTION_FILE,
)