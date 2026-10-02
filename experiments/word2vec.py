import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    RANDOM_STATE,
    TUNING_RESULTS_DIR,
)

from src.data import carregar_dados_treino

from src.evaluation import (
    avaliar_modelo,
)

from src.models import (
    criar_pipeline_word2vec,
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

pipeline = criar_pipeline_word2vec()


# ============================================================
# GRID
# ============================================================

# param_grid = {
#     "embedding__vector_size": [
#         50,
#         100,
#         200,
#     ],

#     "embedding__window": [
#         3,
#         5,
#     ],

#     "embedding__min_count": [
#         2,
#     ],

#     "embedding__epochs": [
#         5,
#     ],

#     "embedding__sg": [
#         0,
#         1,
#     ],

#     "logistic__C": [
#         0.25,
#         0.5,
#         1.0,
#     ],
# }

param_grid = {
    "embedding__vector_size": [
        50,
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
    ],

    "embedding__sg": [
        1,
    ],

    "logistic__C": [
        0.25,
        0.5,
        1.0,
    ],
}

cv_exploratoria = StratifiedKFold(
    n_splits=N_SPLITS_EXPLORATORIA,
    shuffle=True,
    random_state=RANDOM_STATE,
)


grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_exploratoria,
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
print("WORD2VEC")
print("==============================")

print(
    f"Configurações: "
    f"{3 * 3}"
)

print(
    f"Folds de exploração: "
    f"{N_SPLITS_EXPLORATORIA}"
)

grid.fit(X, y)


# ============================================================
# RESULTADOS DA EXPLORAÇÃO
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


print("\nTop configurações:")

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


print("\n==============================")
print("CONFIRMAÇÃO - 5-FOLD")
print("==============================")


print(
    "Melhores parâmetros:"
)

print(
    grid.best_params_
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

resultado = {
    "experiment_id":
        "word2vec_document",

    "experiment_name":
        "Word2Vec + Regressão Logística",

    "category":
        "feature_learning",

    "representation":
        "Word2Vec - média dos embeddings",

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

    "parameters": {
        key: (
            value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value
        in grid.best_params_.items()
    },

    "notes":
        (
            "Word2Vec treinado dentro de cada fold "
            "e representação documental obtida pela "
            "média dos embeddings das palavras."
        ),
}


registrar_experimento(
    resultado
)


arquivo = (
    TUNING_RESULTS_DIR
    / "word2vec.csv"
)


salvar_tabela(
    resultados,
    arquivo,
)


print(
    "\nTabela salva em:",
    arquivo,
)