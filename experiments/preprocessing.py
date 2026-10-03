import pandas as pd
from sklearn.model_selection import cross_validate

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    RANDOM_STATE,
    TUNING_PREPROCESSING_FILE,
)

from src.data import carregar_dados_treino

from src.evaluation import (
    avaliar_modelo,
    criar_cv,
)

from src.models import (
    criar_pipeline_tfidf_word,
)

from src.results import (
    registrar_experimento,
    salvar_tabela,
)

from src.text_preprocessing import (
    STOPWORDS_PT,
    normalizar_texto,
    normalizar_urls,
    normalizar_urls_numeros,
)


# ============================================================
# BASE
# ============================================================

BASE_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 1.0,
    "sublinear_tf": True,
    "C": 0.5,
}


# ============================================================
# CONFIGURAÇÕES DE PRÉ-PROCESSAMENTO
# ============================================================

EXPERIMENTOS = {
    "sem_preprocessamento": {
        "nome": "Sem pré-processamento",
        "preprocessor": None,
        "stop_words": None,
        "strip_accents": None,
    },

    "normalizacao": {
        "nome": "Normalização conservadora",
        "preprocessor": normalizar_texto,
        "stop_words": None,
        "strip_accents": None,
    },

    "stopwords": {
        "nome": "Normalização + stopwords",
        "preprocessor": normalizar_texto,
        "stop_words": list(STOPWORDS_PT),
        "strip_accents": None,
    },

    "urls": {
        "nome": "Normalização + URLs",
        "preprocessor": normalizar_urls,
        "stop_words": None,
        "strip_accents": None,
    },

    "urls_numeros": {
        "nome": "Normalização + URLs + números",
        "preprocessor": normalizar_urls_numeros,
        "stop_words": None,
        "strip_accents": None,
    },

    "urls_numeros_stopwords": {
        "nome": (
            "Normalização + URLs + números + stopwords"
        ),
        "preprocessor": normalizar_urls_numeros,
        "stop_words": list(STOPWORDS_PT),
        "strip_accents": None,
    },
}


# ============================================================
# DADOS
# ============================================================

X, y = carregar_dados_treino()


# ============================================================
# FUNÇÃO PARA CRIAR PIPELINE
# ============================================================

def criar_pipeline_experimento(config):
    return criar_pipeline_tfidf_word(
        **BASE_PARAMS,
        preprocessor=config["preprocessor"],
        stop_words=config["stop_words"],
        strip_accents=config["strip_accents"],
    )


# ============================================================
# EXPERIMENTAÇÃO
# ============================================================

resultados = []

cv_exploratoria = criar_cv(
    N_SPLITS_EXPLORATORIA
)


for experiment_id, config in EXPERIMENTOS.items():

    print("\n==============================")
    print(
        f"PRÉ-PROCESSAMENTO - "
        f"{config['nome'].upper()}"
    )
    print("==============================")


    pipeline = criar_pipeline_experimento(
        config
    )


    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_exploratoria,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
        },
        n_jobs=-1,
        return_train_score=True,
        error_score="raise",
    )


    train_accuracy = (
        scores["train_accuracy"].mean()
    )

    validation_accuracy = (
        scores["test_accuracy"].mean()
    )


    resultado = {
        "experiment_id":
            experiment_id,

        "nome":
            config["nome"],

        "accuracy_3fold":
            validation_accuracy,

        "accuracy_std_3fold":
            scores["test_accuracy"].std(),

        "f1_macro_3fold":
            scores["test_f1_macro"].mean(),

        "train_accuracy_3fold":
            train_accuracy,

        "accuracy_gap_3fold":
            train_accuracy
            - validation_accuracy,
    }


    resultados.append(
        resultado
    )


# ============================================================
# TABELA
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)


resultados_df.sort_values(
    "accuracy_3fold",
    ascending=False,
    inplace=True,
)


resultados_df.reset_index(
    drop=True,
    inplace=True,
)


print("\n==============================")
print("RESULTADOS DOS PRÉ-PROCESSAMENTOS")
print("==============================")


print(
    resultados_df.to_string(
        index=False
    )
)


# ============================================================
# MELHOR CANDIDATO
# ============================================================

melhor_id = (
    resultados_df.iloc[0][
        "experiment_id"
    ]
)

melhor_config = (
    EXPERIMENTOS[melhor_id]
)


melhor_pipeline = (
    criar_pipeline_experimento(
        melhor_config
    )
)


# ============================================================
# CONFIRMAÇÃO 5-FOLD
# ============================================================

_, confirmacao, _ = avaliar_modelo(
    melhor_pipeline,
    X,
    y,
    n_splits=N_SPLITS,
)


print("\n==============================")
print("CONFIRMAÇÃO - 5-FOLD")
print("==============================")


print(
    "Melhor configuração:",
    melhor_config["nome"],
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

resultado_registro = {
    "experiment_id":
        f"preprocessing_{melhor_id}",

    "experiment_name":
        (
            "Pré-processamento - "
            + melhor_config["nome"]
        ),

    "category":
        "preprocessing",

    "representation":
        "TF-IDF palavra",

    "model":
        "Regressão Logística",

    "evaluation":
        "3fold_selection_plus_5fold_confirmation",

    "folds":
        N_SPLITS,

    **confirmacao,

    "selection_accuracy":
        float(
            resultados_df.iloc[0][
                "accuracy_3fold"
            ]
        ),

    "parameters":
        {
            **BASE_PARAMS,
            "preprocessing":
                melhor_id,
        },

    "notes":
        (
            "Comparação exploratória de "
            f"{N_SPLITS_EXPLORATORIA} configurações "
            "de pré-processamento, seguida de "
            f"confirmação em {N_SPLITS} folds."
        ),
}


registrar_experimento(
    resultado_registro
)


salvar_tabela(
    resultados_df,
    TUNING_PREPROCESSING_FILE,
)


print(
    "\nTabela salva em:",
    TUNING_PREPROCESSING_FILE,
)