import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    SENTENCE_TRANSFORMER_BATCH_SIZE,
    SENTENCE_TRANSFORMER_EMBEDDINGS_FILE,
    SENTENCE_TRANSFORMER_MODEL_NAME,
    SENTENCE_TRANSFORMER_NORMALIZE,
    TUNING_SENTENCE_TRANSFORMER_FILE,
)
from src.data import carregar_dados_treino
from src.evaluation import avaliar_modelo, criar_cv
from src.models import criar_regressao_logistica
from src.results import registrar_experimento, salvar_tabela
from src.sentence_embeddings import carregar_ou_gerar_embeddings


# ============================================================
# DADOS
# ============================================================

X, y = carregar_dados_treino()


# ============================================================
# EMBEDDINGS PRÉ-TREINADOS
# ============================================================

embeddings = carregar_ou_gerar_embeddings(
    X,
    model_name=SENTENCE_TRANSFORMER_MODEL_NAME,
    output_file=SENTENCE_TRANSFORMER_EMBEDDINGS_FILE,
    batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
    normalize_embeddings=SENTENCE_TRANSFORMER_NORMALIZE,
)

print("\nShape dos embeddings:", embeddings.shape)


# ============================================================
# CLASSIFICADOR
# ============================================================

pipeline = Pipeline(
    [
        (
            "scale",
            StandardScaler(),
        ),
        (
            "logistic",
            criar_regressao_logistica(
                C=1.0,
            ),
        ),
    ]
)


# ============================================================
# GRID EXPLORATÓRIO
# ============================================================

param_grid = {
    "logistic__C": [
        0.05,
        0.10,
        0.25,
        0.50,
        1.00,
        2.00,
    ],
}


print("\n==============================")
print("SENTENCE TRANSFORMER + LR")
print("==============================")
print("Modelo:", SENTENCE_TRANSFORMER_MODEL_NAME)
print("Configurações:", 6)
print("Folds de exploração:", N_SPLITS_EXPLORATORIA)


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

grid.fit(
    embeddings,
    y,
)


# ============================================================
# RESULTADOS 3-FOLD
# ============================================================

resultados = pd.DataFrame(
    grid.cv_results_
)

resultados = resultados[
    [
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
    embeddings,
    y,
    n_splits=N_SPLITS,
)

melhor_config = {
    "model_name": SENTENCE_TRANSFORMER_MODEL_NAME,
    "normalize_embeddings": SENTENCE_TRANSFORMER_NORMALIZE,
    "embedding_dimension": int(
        embeddings.shape[1]
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
print("Melhor configuração:")
print(melhor_config)

print("\nResultados por fold:")
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
            "tuning_sentence_transformer",

        "experiment_name":
            "Sentence Transformer + Regressão Logística",

        "category":
            "feature_learning",

        "representation":
            "Embedding contextual pré-treinado",

        "model":
            "Regressão Logística",

        "evaluation":
            "3fold_selection_plus_5fold_confirmation",

        "folds":
            N_SPLITS,

        **resumo,

        "selection_accuracy":
            float(
                grid.best_score_
            ),

        "parameters":
            melhor_config,

        "notes": (
            "Embeddings produzidos por um Sentence Transformer "
            "pré-treinado em português brasileiro. O encoder é "
            "congelado; os embeddings são calculados uma vez "
            "antes da validação cruzada e apenas a Regressão "
            "Logística é ajustada nos folds."
        ),
    }
)


salvar_tabela(
    resultados,
    TUNING_SENTENCE_TRANSFORMER_FILE,
)


print(
    "\nTabela salva em:",
    TUNING_SENTENCE_TRANSFORMER_FILE,
)
