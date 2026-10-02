import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    RANDOM_STATE,
    TUNING_TFIDF_WEIGHTED_WORD2VEC_FILE,
    TUNING_HYBRID_TFIDF_WORD2VEC_FILE,
)

from src.data import carregar_dados_treino

from src.evaluation import (
    avaliar_modelo,
    criar_cv,
)

from src.models import (
    criar_pipeline_hybrid_tfidf_word2vec,
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
# MELHOR WORD2VEC
# ============================================================

if not TUNING_TFIDF_WEIGHTED_WORD2VEC_FILE.exists():
    raise FileNotFoundError(
        "O resultado de TF-IDF-weighted Word2Vec não foi encontrado.\n"
        "Execute primeiro:\n"
        "python -m experiments.tfidf_weighted_word2vec"
    )


resultado_weighted = pd.read_csv(
    TUNING_TFIDF_WEIGHTED_WORD2VEC_FILE
)


if resultado_weighted.empty:
    raise ValueError(
        "A tabela de TF-IDF-weighted Word2Vec está vazia."
    )


melhor = resultado_weighted.iloc[0]


W2V_PARAMS = {
    "vector_size": int(
        melhor["vector_size"]
    ),
    "window": int(
        melhor["window"]
    ),
    "min_count": int(
        melhor["min_count"]
    ),
    "epochs": int(
        melhor["epochs"]
    ),
    "sg": int(
        melhor["sg"]
    ),
}


print("\n==============================")
print("HYBRID TF-IDF + WORD2VEC")
print("==============================")


print(
    "\nMelhor Word2Vec encontrado:"
)

print(
    W2V_PARAMS
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = criar_pipeline_hybrid_tfidf_word2vec(
    word_ngram_range=(1, 2),
    word_min_df=5,
    word_max_df=1.0,
    word_sublinear_tf=True,

    vector_size=W2V_PARAMS["vector_size"],
    window=W2V_PARAMS["window"],
    min_count=W2V_PARAMS["min_count"],
    epochs=W2V_PARAMS["epochs"],
    sg=W2V_PARAMS["sg"],

    embedding_tfidf_ngram_range=(1, 1),
    embedding_tfidf_min_df=5,
    embedding_tfidf_max_df=1.0,
    embedding_tfidf_sublinear_tf=True,

    embedding_weight=0.5,
    C=0.5,
)


# ============================================================
# GRID
# ============================================================

param_grid = {
    "features__transformer_weights": [
        {
            "embedding": 0.10,
        },
        {
            "embedding": 0.25,
        },
        {
            "embedding": 0.50,
        },
        {
            "embedding": 1.00,
        },
    ],

    "logistic__C": [
        0.25,
        0.50,
        1.00,
    ],
}


print(
    "\nConfigurações:",
    4 * 3,
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
    n_jobs=2,
    return_train_score=True,
    error_score="raise",
)


grid.fit(
    X,
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
        "param_features__transformer_weights",
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
        "param_features__transformer_weights":
            "transformer_weights",

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


resultados["embedding_weight"] = (
    resultados["transformer_weights"]
    .apply(
        lambda value: value["embedding"]
    )
)


resultados["accuracy_gap_3fold"] = (
    resultados["train_accuracy_3fold"]
    - resultados["accuracy_3fold"]
)


resultados.drop(
    columns=["transformer_weights"],
    inplace=True,
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


melhor_weight = (
    grid.best_params_[
        "features__transformer_weights"
    ]["embedding"]
)


melhor_C = float(
    grid.best_params_[
        "logistic__C"
    ]
)


melhor_config = {
    **W2V_PARAMS,
    "word_ngram_range": (1, 2),
    "word_min_df": 5,
    "word_max_df": 1.0,
    "word_sublinear_tf": True,
    "embedding_tfidf_ngram_range": (1, 1),
    "embedding_tfidf_min_df": 5,
    "embedding_weight": float(
        melhor_weight
    ),
    "C": melhor_C,
}


print("\n==============================")
print("CONFIRMAÇÃO - 5-FOLD")
print("==============================")


print(
    "\nMelhor configuração:"
)

print(
    melhor_config
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
            "tuning_hybrid_tfidf_word2vec",

        "experiment_name":
            "Híbrido TF-IDF + Word2Vec",

        "category":
            "feature_learning",

        "representation":
            "TF-IDF + TF-IDF-weighted Word2Vec",

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
            "Combinação do TF-IDF de palavras com "
            "embeddings Word2Vec ponderados por TF-IDF. "
            "O peso do bloco de embeddings e o C foram "
            "selecionados em 3 folds e confirmados em 5 folds."
        ),
    }
)


salvar_tabela(
    resultados,
    TUNING_HYBRID_TFIDF_WORD2VEC_FILE,
)


print(
    "\nTabela salva em:",
    TUNING_HYBRID_TFIDF_WORD2VEC_FILE,
)