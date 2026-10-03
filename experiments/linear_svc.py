import pandas as pd

from sklearn.model_selection import GridSearchCV

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    TUNING_LINEAR_SVC_FILE,
    TUNING_LINEAR_SVC_FOLDS_FILE,
)
from src.data import carregar_dados_treino
from src.models import criar_pipeline_tfidf_word_svc
from src.evaluation import (
    criar_cv,
    avaliar_modelo,
)
from src.results import (
    registrar_experimento,
    salvar_tabela,
)

# ============================================================
# DADOS
# ============================================================

X, y = carregar_dados_treino()

print("Quantidade de exemplos:", len(X))


# ============================================================
# PIPELINE
# ============================================================

pipeline = criar_pipeline_tfidf_word_svc()


# ============================================================
# GRID PRINCIPAL
# ============================================================

param_grid = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2),
        (1, 3),
    ],

    "tfidf__min_df": [
        1,
        2,
        5,
    ],

    "tfidf__sublinear_tf": [
        False,
        True,
    ],

    "svc__C": [
        0.10,
        0.25,
        0.50,
        1.00,
    ],
}


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


print("\n==============================")
print("GRID FINAL - LINEAR SVC")
print("==============================")

quantidade_configuracoes = (
    len(param_grid["tfidf__ngram_range"])
    * len(param_grid["tfidf__min_df"])
    * len(param_grid["tfidf__sublinear_tf"])
    * len(param_grid["svc__C"])
)

print(
    "Configurações:",
    quantidade_configuracoes,
)

print(
    "Folds de exploração:",
    N_SPLITS_EXPLORATORIA,
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
        "param_tfidf__ngram_range",
        "param_tfidf__min_df",
        "param_tfidf__sublinear_tf",
        "param_svc__C",

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
        "param_tfidf__ngram_range":
            "ngram_range",

        "param_tfidf__min_df":
            "min_df",

        "param_tfidf__sublinear_tf":
            "sublinear_tf",

        "param_svc__C":
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


# ============================================================
# TOP RESULTADOS
# ============================================================

print("\n==============================")
print("TOP 20")
print("==============================")

print(
    resultados.head(20).to_string(
        index=False
    )
)


# ============================================================
# MELHOR CONFIGURAÇÃO
# ============================================================

melhor_config = (
    grid.best_params_
)

print("\n==============================")
print("MELHOR CONFIGURAÇÃO")
print("==============================")

print(
    melhor_config
)

print(
    f"Accuracy 3-fold: "
    f"{grid.best_score_:.4f}"
)


# ============================================================
# CONFIRMAÇÃO 5-FOLD
# ============================================================

melhor_pipeline = (
    grid.best_estimator_
)

_, confirmacao, folds = avaliar_modelo(
    melhor_pipeline,
    X,
    y,
    n_splits=N_SPLITS,
)


print("\n==============================")
print("CONFIRMAÇÃO - 5-FOLD")
print("==============================")

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


print("\nResultados por fold:")

print(
    folds.to_string(
        index=False
    )
)


# ============================================================
# REGISTRO
# ============================================================

resultado = {
    "experiment_id":
        "linear_svc_grid",

    "experiment_name":
        "Grid - TF-IDF + LinearSVC",

    "category":
        "linearsvc",

    "representation":
        "TF-IDF palavra",

    "model":
        "LinearSVC",

    "evaluation":
        (
            "3fold_selection_plus_"
            "5fold_confirmation"
        ),

    "folds":
        N_SPLITS,

    **confirmacao,

    "selection_accuracy":
        float(grid.best_score_),

    "parameters":
        melhor_config,

    "notes":
        (
            "Busca ampliada de TF-IDF + LinearSVC. "
            "Foram testados n-gramas de palavras, "
            "min_df, sublinear_tf e C."
        ),
}


registrar_experimento(
    resultado
)


# ============================================================
# SALVAMENTO
# ============================================================

salvar_tabela(
    resultados,
    TUNING_LINEAR_SVC_FILE,
)

salvar_tabela(
    folds,
    TUNING_LINEAR_SVC_FOLDS_FILE,
)


print(
    "\nResultados salvos."
)