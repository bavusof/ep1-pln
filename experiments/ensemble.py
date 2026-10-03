import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    CLASSES,
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    RANDOM_STATE,
    SENTENCE_TRANSFORMER_BATCH_SIZE,
    SENTENCE_TRANSFORMER_EMBEDDINGS_FILE,
    SENTENCE_TRANSFORMER_MODEL_NAME,
    SENTENCE_TRANSFORMER_NORMALIZE,
)

from src.data import carregar_dados_treino
from src.evaluation import criar_cv
from src.results import (
    registrar_experimento,
    salvar_tabela,
)

from src.sentence_embeddings import (
    carregar_ou_gerar_embeddings,
)


# ============================================================
# CONFIGURAÇÕES DOS DOIS MODELOS
# ============================================================

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_df": 1.0,
    "sublinear_tf": True,
}

C_TFIDF = 1.0

C_HYBRID = 0.5

EMBEDDING_WEIGHT = 0.1


# Pesos testados para a combinação.
BLEND_WEIGHTS = [
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    1.0,
]


def probabilidades_na_ordem(
    modelo,
    probabilidades,
):
    """
    Garante que as probabilidades estejam na ordem:

        c1, c234, c5
    """

    resultado = np.zeros(
        (
            len(probabilidades),
            len(CLASSES),
        )
    )

    for i, classe in enumerate(
        modelo.classes_
    ):
        indice = CLASSES.index(classe)

        resultado[:, indice] = (
            probabilidades[:, i]
        )

    return resultado


def avaliar_blend(
    X,
    y,
    embeddings,
    weight,
    n_splits,
):
    """
    Avalia um peso de combinação usando validação cruzada.

    weight = 0:
        apenas TF-IDF

    weight = 1:
        apenas híbrido

    valores intermediários:
        combinação dos dois.
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

        # ====================================================
        # MODELO A - TF-IDF
        # ====================================================

        tfidf = TfidfVectorizer(
            **TFIDF_PARAMS
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

        modelo_tfidf = LogisticRegression(
            C=C_TFIDF,
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

        modelo_tfidf.fit(
            X_train_tfidf,
            y_train,
        )

        proba_tfidf = (
            modelo_tfidf.predict_proba(
                X_valid_tfidf
            )
        )

        proba_tfidf = (
            probabilidades_na_ordem(
                modelo_tfidf,
                proba_tfidf,
            )
        )

        # ====================================================
        # MODELO B - TF-IDF + SENTENCE TRANSFORMER
        # ====================================================

        embedding_train = csr_matrix(
            embeddings[
                train_idx
            ]
            * EMBEDDING_WEIGHT
        )

        embedding_valid = csr_matrix(
            embeddings[
                valid_idx
            ]
            * EMBEDDING_WEIGHT
        )

        X_train_hybrid = hstack(
            [
                X_train_tfidf,
                embedding_train,
            ],
            format="csr",
        )

        X_valid_hybrid = hstack(
            [
                X_valid_tfidf,
                embedding_valid,
            ],
            format="csr",
        )

        modelo_hybrid = LogisticRegression(
            C=C_HYBRID,
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

        modelo_hybrid.fit(
            X_train_hybrid,
            y_train,
        )

        proba_hybrid = (
            modelo_hybrid.predict_proba(
                X_valid_hybrid
            )
        )

        proba_hybrid = (
            probabilidades_na_ordem(
                modelo_hybrid,
                proba_hybrid,
            )
        )

        # ====================================================
        # COMBINAÇÃO
        # ====================================================

        proba_final = (
            (1.0 - weight)
            * proba_tfidf
            +
            weight
            * proba_hybrid
        )

        pred_final = np.array(
            CLASSES
        )[
            np.argmax(
                proba_final,
                axis=1,
            )
        ]

        resultados.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(
                    y_valid,
                    pred_final,
                ),
                "f1_macro": f1_score(
                    y_valid,
                    pred_final,
                    labels=CLASSES,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

    df = pd.DataFrame(
        resultados
    )

    return {
        "accuracy": float(
            df["accuracy"].mean()
        ),
        "accuracy_std": float(
            df["accuracy"].std(
                ddof=0
            )
        ),
        "f1_macro": float(
            df["f1_macro"].mean()
        ),
        "f1_macro_std": float(
            df["f1_macro"].std(
                ddof=0
            )
        ),
    }


def main():

    # ========================================================
    # DADOS
    # ========================================================

    X, y = carregar_dados_treino()

    embeddings = (
        carregar_ou_gerar_embeddings(
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
    )

    # ========================================================
    # EXPLORAÇÃO - 3 FOLD
    # ========================================================

    resultados = []

    print("\n==============================")
    print("ENSEMBLE - EXPLORAÇÃO")
    print("==============================")

    for weight in BLEND_WEIGHTS:

        print(
            f"\nPeso híbrido: {weight:.2f}"
        )

        metricas = avaliar_blend(
            X,
            y,
            embeddings,
            weight,
            N_SPLITS_EXPLORATORIA,
        )

        resultados.append(
            {
                "embedding_weight":
                    weight,

                "accuracy_3fold":
                    metricas["accuracy"],

                "accuracy_std_3fold":
                    metricas["accuracy_std"],

                "f1_macro_3fold":
                    metricas["f1_macro"],
            }
        )

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

    print("\nResultados:")
    print(
        resultados_df.to_string(
            index=False
        )
    )

    # ========================================================
    # MELHOR CONFIGURAÇÃO
    # ========================================================

    melhor_weight = float(
        resultados_df.iloc[0][
            "embedding_weight"
        ]
    )

    melhor_accuracy_3fold = float(
        resultados_df.iloc[0][
            "accuracy_3fold"
        ]
    )

    print(
        "\nMelhor peso:",
        melhor_weight,
    )

    # ========================================================
    # CONFIRMAÇÃO - 5 FOLD
    # ========================================================

    confirmacao = avaliar_blend(
        X,
        y,
        embeddings,
        melhor_weight,
        N_SPLITS,
    )

    print("\n==============================")
    print("CONFIRMAÇÃO - 5 FOLD")
    print("==============================")

    print(
        f"Peso híbrido: "
        f"{melhor_weight:.2f}"
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

    # ========================================================
    # RESULTADO
    # ========================================================

    registrar_experimento(
        {
            "experiment_id":
                "ensemble_tfidf_sentence_transformer",

            "experiment_name":
                "Ensemble TF-IDF + Sentence Transformer",

            "category":
                "ensemble",

            "representation":
                (
                    "TF-IDF palavra + "
                    "embedding contextual"
                ),

            "model":
                "Média ponderada de probabilidades",

            "evaluation":
                (
                    "3fold_selection_plus_"
                    "5fold_confirmation"
                ),

            "folds":
                N_SPLITS,

            "accuracy":
                confirmacao["accuracy"],

            "accuracy_std":
                confirmacao["accuracy_std"],

            "f1_macro":
                confirmacao["f1_macro"],

            "f1_macro_std":
                confirmacao["f1_macro_std"],

            "selection_accuracy":
                melhor_accuracy_3fold,

            "parameters":
                {
                    "tfidf":
                        TFIDF_PARAMS,

                    "C_tfidf":
                        C_TFIDF,

                    "C_hybrid":
                        C_HYBRID,

                    "embedding_weight":
                        melhor_weight,

                    "model_name":
                        SENTENCE_TRANSFORMER_MODEL_NAME,
                },

            "notes":
                (
                    "Combinação das probabilidades "
                    "de um modelo TF-IDF + LR e de "
                    "um modelo híbrido TF-IDF + "
                    "Sentence Transformer."
                ),
        }
    )


if __name__ == "__main__":
    main()