import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid

from src.config import (
    CLASSES,
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    SENTENCE_TRANSFORMER_BATCH_SIZE,
    SENTENCE_TRANSFORMER_EMBEDDINGS_FILE,
    SENTENCE_TRANSFORMER_MODEL_NAME,
    SENTENCE_TRANSFORMER_NORMALIZE,
    TUNING_HYBRID_TFIDF_SENTENCE_TRANSFORMER_FILE,
    TUNING_HYBRID_TFIDF_SENTENCE_TRANSFORMER_FOLDS_FILE,
)

from src.data import carregar_dados_treino

from src.evaluation import criar_cv

from src.models import criar_regressao_logistica

from src.results import (
    registrar_experimento,
    salvar_tabela,
)

from src.sentence_embeddings import (
    carregar_ou_gerar_embeddings,
)


# ============================================================
# TF-IDF BASE
# ============================================================

TFIDF_BASE_PARAMS = {
    "max_df": 1.0,
}


# ============================================================
# GRID
# ============================================================

PARAM_GRID = {
    "embedding_weight": [
        # 0.1,
        0.2,
        # 0.3,
        0.4,
        0.6,
        # 0.7,
        0.8,
        # 0.9,
        # 1.0,
    ],

    "C": [
        0.25,
        0.50,
        1.00,
    ],

    "ngram_range": [
        # (1, 1),
        (1, 2),
    ],

    "min_df": [
        1,
        5,
    ],

    "sublinear_tf": [
        False,
        True,
    ],
}

def calcular_metricas(y_true, y_pred):
    """Calcula as métricas utilizadas no projeto."""
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


def avaliar_configuracao(
    X,
    y,
    embeddings,
    config,
    n_splits,
):
    """
    Avalia uma configuração usando validação cruzada.

    O TF-IDF é ajustado exclusivamente no conjunto de treino
    de cada fold.

    Os embeddings do Sentence Transformer já são pré-treinados
    e congelados, portanto podem ser reutilizados diretamente.
    """

    cv = criar_cv(
        n_splits
    )

    resultados_folds = []

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

        # ====================================================
        # TF-IDF
        # ====================================================

        tfidf = TfidfVectorizer(
            ngram_range=tuple(
                config["ngram_range"]
            ),
            min_df=int(
                config["min_df"]
            ),
            max_df=1.0,
            sublinear_tf=bool(
                config["sublinear_tf"]
            ),
        )

        X_train_tfidf = (
            tfidf.fit_transform(
                X_train
            )
        )

        X_valid_tfidf = (
            tfidf.transform(
                X_valid
            )
        )

        # ====================================================
        # EMBEDDINGS
        # ====================================================

        embedding_weight = float(
            config[
                "embedding_weight"
            ]
        )

        X_train_embedding = csr_matrix(
            embeddings[train_idx]
            * embedding_weight
        )

        X_valid_embedding = csr_matrix(
            embeddings[valid_idx]
            * embedding_weight
        )

        # ====================================================
        # CONCATENAÇÃO
        # ====================================================

        X_train_hybrid = hstack(
            [
                X_train_tfidf,
                X_train_embedding,
            ],
            format="csr",
        )

        X_valid_hybrid = hstack(
            [
                X_valid_tfidf,
                X_valid_embedding,
            ],
            format="csr",
        )

        # ====================================================
        # CLASSIFICADOR
        # ====================================================

        modelo = criar_regressao_logistica(
            C=float(
                config["C"]
            )
        )

        modelo.fit(
            X_train_hybrid,
            y_train,
        )

        # ====================================================
        # PREDIÇÕES
        # ====================================================

        pred_train = modelo.predict(
            X_train_hybrid
        )

        pred_valid = modelo.predict(
            X_valid_hybrid
        )

        # ====================================================
        # MÉTRICAS
        # ====================================================

        metricas_train = calcular_metricas(
            y_train,
            pred_train,
        )

        metricas_valid = calcular_metricas(
            y_valid,
            pred_valid,
        )

        resultados_folds.append(
            {
                "fold": fold,

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

    df_folds = pd.DataFrame(
        resultados_folds
    )

    resumo = {
        "accuracy":
            float(
                df_folds[
                    "accuracy"
                ].mean()
            ),

        "accuracy_std":
            float(
                df_folds[
                    "accuracy"
                ].std(
                    ddof=0
                )
            ),

        "f1_macro":
            float(
                df_folds[
                    "f1_macro"
                ].mean()
            ),

        "f1_macro_std":
            float(
                df_folds[
                    "f1_macro"
                ].std(
                    ddof=0
                )
            ),

        "train_accuracy":
            float(
                df_folds[
                    "train_accuracy"
                ].mean()
            ),

        "train_accuracy_std":
            float(
                df_folds[
                    "train_accuracy"
                ].std(
                    ddof=0
                )
            ),

        "train_f1_macro":
            float(
                df_folds[
                    "train_f1_macro"
                ].mean()
            ),

        "train_f1_macro_std":
            float(
                df_folds[
                    "train_f1_macro"
                ].std(
                    ddof=0
                )
            ),
    }

    resumo["accuracy_gap"] = (
        resumo["train_accuracy"]
        - resumo["accuracy"]
    )

    return resumo, df_folds


def main():

    # ========================================================
    # DADOS
    # ========================================================

    X, y = carregar_dados_treino()

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    embeddings = carregar_ou_gerar_embeddings(
        X,
        model_name=(
            SENTENCE_TRANSFORMER_MODEL_NAME
        ),
        output_file=(
            SENTENCE_TRANSFORMER_EMBEDDINGS_FILE
        ),
        batch_size=(
            SENTENCE_TRANSFORMER_BATCH_SIZE
        ),
        normalize_embeddings=(
            SENTENCE_TRANSFORMER_NORMALIZE
        ),
    )

    if embeddings.shape[0] != len(X):
        raise ValueError(
            "Quantidade de embeddings diferente "
            "da quantidade de documentos."
        )

    print("\n==============================")
    print(
        "TF-IDF + SENTENCE TRANSFORMER"
    )
    print("==============================")

    print(
        "Modelo:",
        SENTENCE_TRANSFORMER_MODEL_NAME,
    )

    print(
        "Dimensão dos embeddings:",
        embeddings.shape[1],
    )

    configuracoes = list(
        ParameterGrid(
            PARAM_GRID
        )
    )

    print(
        "Configurações:",
        len(configuracoes),
    )

    print(
        "Folds de exploração:",
        N_SPLITS_EXPLORATORIA,
    )

    # ========================================================
    # EXPLORAÇÃO 3-FOLD
    # ========================================================

    resultados = []

    melhor_config = None
    melhor_accuracy = -np.inf

    for indice, config in enumerate(
        configuracoes,
        start=1,
    ):

        print(
            "\n------------------------------"
        )

        print(
            f"Configuração "
            f"{indice}/{len(configuracoes)}"
        )

        print(
            "Embedding weight:",
            config["embedding_weight"],
        )

        print(
            "C:",
            config["C"],
        )

        resumo, _ = avaliar_configuracao(
            X,
            y,
            embeddings,
            config,
            N_SPLITS_EXPLORATORIA,
        )

        resultados.append(
            {
                "embedding_weight":
                    config[
                        "embedding_weight"
                    ],

                "C":
                    config["C"],

                "train_accuracy_3fold":
                    resumo[
                        "train_accuracy"
                    ],

                "train_accuracy_std_3fold":
                    resumo[
                        "train_accuracy_std"
                    ],

                "accuracy_3fold":
                    resumo[
                        "accuracy"
                    ],

                "accuracy_std_3fold":
                    resumo[
                        "accuracy_std"
                    ],

                "train_f1_macro_3fold":
                    resumo[
                        "train_f1_macro"
                    ],

                "f1_macro_3fold":
                    resumo[
                        "f1_macro"
                    ],

                "accuracy_gap_3fold":
                    resumo[
                        "accuracy_gap"
                    ],
            }
        )

        print(
            f"Accuracy 3-fold: "
            f"{resumo['accuracy']:.4f}"
        )

        if (
            resumo["accuracy"]
            > melhor_accuracy
        ):
            melhor_accuracy = (
                resumo["accuracy"]
            )

            melhor_config = (
                config.copy()
            )

    tabela = pd.DataFrame(
        resultados
    )

    tabela.sort_values(
        "accuracy_3fold",
        ascending=False,
        inplace=True,
    )

    tabela.reset_index(
        drop=True,
        inplace=True,
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
        tabela.to_string(
            index=False
        )
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

    print(
        "Melhor configuração:",
        melhor_config,
    )

    resumo_5fold, folds_5fold = (
        avaliar_configuracao(
            X,
            y,
            embeddings,
            melhor_config,
            N_SPLITS,
        )
    )

    print(
        "\nResultados por fold:"
    )

    print(
        folds_5fold.to_string(
            index=False
        )
    )

    print(
        f"\nAccuracy: "
        f"{resumo_5fold['accuracy']:.4f} "
        f"± "
        f"{resumo_5fold['accuracy_std']:.4f}"
    )

    print(
        f"F1 Macro: "
        f"{resumo_5fold['f1_macro']:.4f} "
        f"± "
        f"{resumo_5fold['f1_macro_std']:.4f}"
    )

    print(
        f"Accuracy treino: "
        f"{resumo_5fold['train_accuracy']:.4f}"
    )

    print(
        f"Gap: "
        f"{resumo_5fold['accuracy_gap']:.4f}"
    )

    # ========================================================
    # REGISTRO CONSOLIDADO
    # ========================================================

    registrar_experimento(
        {
            "experiment_id":
                "tuning_hybrid_tfidf_sentence_transformer",

            "experiment_name":
                "TF-IDF + Sentence Transformer",

            "category":
                "feature_learning",

            "representation":
                (
                    "TF-IDF palavra + "
                    "embedding contextual "
                    "pré-treinado"
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

            **resumo_5fold,

            "selection_accuracy":
                float(
                    melhor_accuracy
                ),

            "parameters":
                {
                    "model_name":
                        SENTENCE_TRANSFORMER_MODEL_NAME,

                    "normalize_embeddings":
                        SENTENCE_TRANSFORMER_NORMALIZE,

                    "embedding_dimension":
                        int(
                            embeddings.shape[1]
                        ),

                    "embedding_weight":
                        float(
                            melhor_config[
                                "embedding_weight"
                            ]
                        ),

                    "ngram_range": melhor_config["ngram_range"],

                    "min_df": int(melhor_config["min_df"]),
                    
                    "sublinear_tf": bool(
                        melhor_config["sublinear_tf"]
                    ),

                    "C":
                        float(
                            melhor_config["C"]
                        ),
                },

            "notes":
                (
                    "Híbrido TF-IDF + embedding "
                    "de Sentence Transformer. "
                    "O encoder é pré-treinado e "
                    "congelado. Os embeddings são "
                    "calculados uma única vez. "
                    "O TF-IDF é ajustado dentro "
                    "de cada fold. As duas "
                    "representações são "
                    "concatenadas antes da "
                    "Regressão Logística."
                ),
        }
    )

    # ========================================================
    # SALVAR
    # ========================================================

    salvar_tabela(
        tabela,
        TUNING_HYBRID_TFIDF_SENTENCE_TRANSFORMER_FILE,
    )

    salvar_tabela(
        folds_5fold,
        TUNING_HYBRID_TFIDF_SENTENCE_TRANSFORMER_FOLDS_FILE,
    )

    print(
        "\nTabela 3-fold salva em:"
    )

    print(
        TUNING_HYBRID_TFIDF_SENTENCE_TRANSFORMER_FILE
    )

    print(
        "\nFolds 5-fold salvos em:"
    )

    print(
        TUNING_HYBRID_TFIDF_SENTENCE_TRANSFORMER_FOLDS_FILE
    )


if __name__ == "__main__":
    main()