import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASSES,
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    SENTENCE_TRANSFORMER_MODEL_NAME,
    TOKEN_WEIGHTED_BERT_BATCH_SIZE,
    TOKEN_WEIGHTED_BERT_LENGTHS,
    TUNING_TOKEN_WEIGHTED_BERT_FILE,
    TUNING_TOKEN_WEIGHTED_BERT_FOLDS_FILE,
)

from src.data import carregar_dados_treino

from src.evaluation import criar_cv

from src.models import criar_regressao_logistica

from src.results import (
    registrar_experimento,
    salvar_tabela,
)

from src.token_weighted_bert import (
    carregar_modelo_token_bert,
    gerar_embeddings_token_weighted,
    tokenizar_documentos,
)


# ============================================================
# TF-IDF
# ============================================================

TFIDF_PARAMS = {
    "lowercase": False,
    "analyzer": lambda tokens: tokens,
    "token_pattern": None,
    "min_df": 5,
    "max_df": 1.0,
    "sublinear_tf": True,
}


# ============================================================
# CLASSIFICADOR
# ============================================================

C_VALUES = [
    0.25,
    0.50,
    1.00,
]


def calcular_metricas(
    y_true,
    y_pred,
):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
    }


def avaliar_comprimento(
    X,
    y,
    model,
    tokenizer,
    transformer,
    max_length,
    n_splits,
    batch_size,
):
    """
    Avalia um max_length em validação cruzada.

    O TF-IDF é ajustado novamente em cada fold.
    O BERT permanece congelado.
    """

    cv = criar_cv(
        n_splits
    )

    resultados = []

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_valid = X.iloc[
            valid_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_valid = y.iloc[
            valid_idx
        ]

        print(
            f"\n[{max_length} tokens] "
            f"Fold {fold}/{n_splits}: tokenização..."
        )

        # ----------------------------------------------------
        # TOKENIZAÇÃO
        # ----------------------------------------------------

        (
            train_tokens,
            train_ids,
            train_masks,
        ) = tokenizar_documentos(
            X_train.tolist(),
            tokenizer,
            max_length,
        )

        (
            valid_tokens,
            valid_ids,
            valid_masks,
        ) = tokenizar_documentos(
            X_valid.tolist(),
            tokenizer,
            max_length,
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        tfidf = TfidfVectorizer(
            **TFIDF_PARAMS
        )

        tfidf.fit(
            train_tokens
        )

        # ----------------------------------------------------
        # BERT + POOLING PONDERADO
        # ----------------------------------------------------

        print(
            f"[{max_length} tokens] "
            f"Fold {fold}: embeddings treino..."
        )

        train_embeddings = (
            gerar_embeddings_token_weighted(
                train_tokens,
                train_ids,
                train_masks,
                tfidf,
                transformer,
                tokenizer,
                batch_size=batch_size,
            )
        )

        print(
            f"[{max_length} tokens] "
            f"Fold {fold}: embeddings validação..."
        )

        valid_embeddings = (
            gerar_embeddings_token_weighted(
                valid_tokens,
                valid_ids,
                valid_masks,
                tfidf,
                transformer,
                tokenizer,
                batch_size=batch_size,
            )
        )

        # ----------------------------------------------------
        # STANDARD SCALER
        # ----------------------------------------------------

        scaler = StandardScaler()

        train_embeddings = (
            scaler.fit_transform(
                train_embeddings
            )
        )

        valid_embeddings = (
            scaler.transform(
                valid_embeddings
            )
        )

        # ----------------------------------------------------
        # DIFERENTES C
        # ----------------------------------------------------

        for C in C_VALUES:

            modelo = criar_regressao_logistica(
                C=C
            )

            modelo.fit(
                train_embeddings,
                y_train,
            )

            pred_train = modelo.predict(
                train_embeddings
            )

            pred_valid = modelo.predict(
                valid_embeddings
            )

            metricas_train = (
                calcular_metricas(
                    y_train,
                    pred_train,
                )
            )

            metricas_valid = (
                calcular_metricas(
                    y_valid,
                    pred_valid,
                )
            )

            resultados.append(
                {
                    "max_length":
                        max_length,

                    "C":
                        C,

                    "fold":
                        fold,

                    "train_accuracy":
                        metricas_train[
                            "accuracy"
                        ],

                    "accuracy":
                        metricas_valid[
                            "accuracy"
                        ],

                    "train_f1_macro":
                        metricas_train[
                            "f1_macro"
                        ],

                    "f1_macro":
                        metricas_valid[
                            "f1_macro"
                        ],
                }
            )

    return pd.DataFrame(
        resultados
    )


def resumir_resultados(
    df_folds,
):
    """
    Consolida os folds por configuração.
    """

    resultados = (
        df_folds
        .groupby(
            [
                "max_length",
                "C",
            ],
            as_index=False,
        )
        .agg(
            train_accuracy_3fold=(
                "train_accuracy",
                "mean",
            ),
            train_accuracy_std_3fold=(
                "train_accuracy",
                "std",
            ),
            accuracy_3fold=(
                "accuracy",
                "mean",
            ),
            accuracy_std_3fold=(
                "accuracy",
                "std",
            ),
            train_f1_macro_3fold=(
                "train_f1_macro",
                "mean",
            ),
            f1_macro_3fold=(
                "f1_macro",
                "mean",
            ),
        )
    )

    resultados[
        "accuracy_gap_3fold"
    ] = (
        resultados[
            "train_accuracy_3fold"
        ]
        - resultados[
            "accuracy_3fold"
        ]
    )

    return (
        resultados
        .sort_values(
            "accuracy_3fold",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def main():

    # ========================================================
    # DADOS
    # ========================================================

    X, y = carregar_dados_treino()

    print(
        "\n=============================="
    )

    print(
        "TOKEN-LEVEL TF-IDF-WEIGHTED BERT"
    )

    print(
        "=============================="
    )

    print(
        "Modelo:",
        SENTENCE_TRANSFORMER_MODEL_NAME,
    )

    print(
        "Max lengths:",
        TOKEN_WEIGHTED_BERT_LENGTHS,
    )

    print(
        "C:",
        C_VALUES,
    )

    print(
        "Folds de exploração:",
        N_SPLITS_EXPLORATORIA,
    )

    # ========================================================
    # BERT
    # ========================================================

    (
        model,
        tokenizer,
        transformer,
    ) = carregar_modelo_token_bert(
        SENTENCE_TRANSFORMER_MODEL_NAME
    )

    print(
        "Device:",
        model.device,
    )

    # ========================================================
    # EXPLORAÇÃO 3-FOLD
    # ========================================================

    resultados_folds = []

    for max_length in (
        TOKEN_WEIGHTED_BERT_LENGTHS
    ):

        df_folds = avaliar_comprimento(
            X,
            y,
            model,
            tokenizer,
            transformer,
            max_length,
            N_SPLITS_EXPLORATORIA,
            TOKEN_WEIGHTED_BERT_BATCH_SIZE,
        )

        resultados_folds.append(
            df_folds
        )

    folds_3 = pd.concat(
        resultados_folds,
        ignore_index=True,
    )

    resultados_3fold = (
        resumir_resultados(
            folds_3
        )
    )

    print(
        "\n=============================="
    )

    print(
        "RESULTADOS - 3-FOLD"
    )

    print(
        "=============================="
    )

    print(
        resultados_3fold.to_string(
            index=False
        )
    )

    # ========================================================
    # MELHOR CONFIGURAÇÃO
    # ========================================================

    melhor = (
        resultados_3fold.iloc[0]
    )

    melhor_config = {
        "max_length":
            int(
                melhor[
                    "max_length"
                ]
            ),

        "C":
            float(
                melhor["C"]
            ),
    }

    print(
        "\nMelhor configuração:",
        melhor_config,
    )

    # ========================================================
    # CONFIRMAÇÃO 5-FOLD
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "CONFIRMAÇÃO - 5-FOLD"
    )

    print(
        "=============================="
    )

    folds_5 = avaliar_comprimento(
        X,
        y,
        model,
        tokenizer,
        transformer,
        melhor_config[
            "max_length"
        ],
        N_SPLITS,
        TOKEN_WEIGHTED_BERT_BATCH_SIZE,
    )

    folds_5 = folds_5[
        folds_5["C"]
        == melhor_config["C"]
    ].copy()

    resumo_5 = {
        "accuracy":
            float(
                folds_5[
                    "accuracy"
                ].mean()
            ),

        "accuracy_std":
            float(
                folds_5[
                    "accuracy"
                ].std(
                    ddof=0
                )
            ),

        "f1_macro":
            float(
                folds_5[
                    "f1_macro"
                ].mean()
            ),

        "f1_macro_std":
            float(
                folds_5[
                    "f1_macro"
                ].std(
                    ddof=0
                )
            ),

        "train_accuracy":
            float(
                folds_5[
                    "train_accuracy"
                ].mean()
            ),

        "train_accuracy_std":
            float(
                folds_5[
                    "train_accuracy"
                ].std(
                    ddof=0
                )
            ),

        "train_f1_macro":
            float(
                folds_5[
                    "train_f1_macro"
                ].mean()
            ),

        "train_f1_macro_std":
            float(
                folds_5[
                    "train_f1_macro"
                ].std(
                    ddof=0
                )
            ),
    }

    resumo_5[
        "accuracy_gap"
    ] = (
        resumo_5[
            "train_accuracy"
        ]
        - resumo_5[
            "accuracy"
        ]
    )

    print(
        "\nResultados 5-fold:"
    )

    print(
        folds_5.to_string(
            index=False
        )
    )

    print(
        f"\nAccuracy: "
        f"{resumo_5['accuracy']:.4f} "
        f"± "
        f"{resumo_5['accuracy_std']:.4f}"
    )

    print(
        f"F1 Macro: "
        f"{resumo_5['f1_macro']:.4f} "
        f"± "
        f"{resumo_5['f1_macro_std']:.4f}"
    )

    print(
        f"Accuracy treino: "
        f"{resumo_5['train_accuracy']:.4f}"
    )

    print(
        f"Gap: "
        f"{resumo_5['accuracy_gap']:.4f}"
    )

    # ========================================================
    # REGISTRO CONSOLIDADO
    # ========================================================

    registrar_experimento(
        {
            "experiment_id":
                "tuning_token_weighted_bert",

            "experiment_name":
                "Token-level TF-IDF-weighted BERT",

            "category":
                "feature_learning",

            "representation":
                (
                    "BERT token embeddings + "
                    "TF-IDF weighted pooling"
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

            **resumo_5,

            "selection_accuracy":
                float(
                    melhor[
                        "accuracy_3fold"
                    ]
                ),

            "parameters":
                {
                    "model_name":
                        SENTENCE_TRANSFORMER_MODEL_NAME,

                    "max_length":
                        melhor_config[
                            "max_length"
                        ],

                    "C":
                        melhor_config["C"],

                    "tfidf":
                        {
                            "min_df": 5,
                            "max_df": 1.0,
                            "sublinear_tf": True,
                            "representation":
                                "BERT subwords",
                        },
                },

            "notes":
                (
                    "Encoder contextual pré-treinado "
                    "e congelado. TF-IDF ajustado "
                    "exclusivamente no treino de cada "
                    "fold. Cada documento é representado "
                    "por pooling dos embeddings "
                    "contextuais dos tokens ponderado "
                    "pelos pesos TF-IDF dos subwords."
                ),
        }
    )

    # ========================================================
    # SALVAR
    # ========================================================

    salvar_tabela(
        resultados_3fold,
        TUNING_TOKEN_WEIGHTED_BERT_FILE,
    )

    salvar_tabela(
        folds_5,
        TUNING_TOKEN_WEIGHTED_BERT_FOLDS_FILE,
    )

    print(
        "\nResultado 3-fold salvo em:"
    )

    print(
        TUNING_TOKEN_WEIGHTED_BERT_FILE
    )

    print(
        "\nFolds 5-fold salvos em:"
    )

    print(
        TUNING_TOKEN_WEIGHTED_BERT_FOLDS_FILE
    )


if __name__ == "__main__":
    main()