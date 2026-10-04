from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from experiments.ensemble import (
    C_HYBRID,
    C_TFIDF,
    EMBEDDING_WEIGHT,
    TFIDF_PARAMS,
    probabilidades_na_ordem,
)
from src.config import (
    CLASSES,
    DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    ROOT_DIR,
    SENTENCE_TRANSFORMER_BATCH_SIZE,
    SENTENCE_TRANSFORMER_EMBEDDINGS_FILE,
    SENTENCE_TRANSFORMER_MODEL_NAME,
    SENTENCE_TRANSFORMER_NORMALIZE,
)
from src.data import carregar_dados_treino
from src.sentence_embeddings import carregar_ou_gerar_embeddings


TEST_FILE = DATA_DIR / "test1.xlsx"
OUTPUT_FILE = ROOT_DIR / "test1_rotulado.xlsx"

# Peso final do ensemble:
# 20% TF-IDF puro + 80% modelo híbrido
FINAL_HYBRID_BLEND_WEIGHT = 0.8


def _hash_textos(textos: pd.Series) -> str:
    """
    Gera um identificador para o arquivo de teste.

    Isso evita reutilizar por engano embeddings de outro arquivo
    de teste que tenha a mesma quantidade de linhas.
    """

    digest = hashlib.sha256()

    for texto in textos:
        digest.update(
            str(texto).encode(
                "utf-8",
                errors="replace",
            )
        )
        digest.update(b"\0")

    return digest.hexdigest()[:12]


def carregar_teste() -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega o conjunto de teste e valida a presença
    da coluna resp_text.
    """

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de teste não encontrado: {TEST_FILE}\n"
            "Coloque test1.xlsx dentro da pasta data/."
        )

    df = pd.read_excel(TEST_FILE)

    if "resp_text" not in df.columns:
        raise ValueError(
            "O arquivo data/test1.xlsx precisa conter "
            "a coluna 'resp_text'."
        )

    textos = (
        df["resp_text"]
        .fillna("")
        .astype(str)
    )

    if len(textos) == 0:
        raise ValueError(
            "O arquivo de teste está vazio."
        )

    return df, textos


def main() -> None:
    # ========================================================
    # DADOS
    # ========================================================

    X_train, y_train = carregar_dados_treino()
    _, X_test = carregar_teste()

    print(
        f"Treino: {len(X_train)} exemplos"
    )

    print(
        f"Teste: {len(X_test)} exemplos"
    )

    print(
        f"Classes: {CLASSES}"
    )

    # ========================================================
    # EMBEDDINGS DO SENTENCE TRANSFORMER
    # ========================================================

    embeddings_train = (
        carregar_ou_gerar_embeddings(
            X_train,
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

    test_hash = _hash_textos(
        X_test
    )

    test_embeddings_file = (
        RESULTS_DIR
        / "embeddings"
        / (
            "bertimbau_base_portuguese_sts_"
            f"test_{test_hash}.npy"
        )
    )

    embeddings_test = (
        carregar_ou_gerar_embeddings(
            X_test,
            model_name=(
                SENTENCE_TRANSFORMER_MODEL_NAME
            ),
            output_file=(
                test_embeddings_file
            ),
            batch_size=(
                SENTENCE_TRANSFORMER_BATCH_SIZE
            ),
            normalize_embeddings=(
                SENTENCE_TRANSFORMER_NORMALIZE
            ),
        )
    )

    if (
        embeddings_train.shape[1]
        != embeddings_test.shape[1]
    ):
        raise RuntimeError(
            "Dimensão dos embeddings de treino e teste "
            "não coincide: "
            f"{embeddings_train.shape[1]} != "
            f"{embeddings_test.shape[1]}"
        )

    # ========================================================
    # MODELO A
    # TF-IDF + REGRESSÃO LOGÍSTICA
    # ========================================================

    tfidf = TfidfVectorizer(
        **TFIDF_PARAMS
    )

    X_train_tfidf = (
        tfidf.fit_transform(
            X_train
        )
    )

    X_test_tfidf = (
        tfidf.transform(
            X_test
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
        probabilidades_na_ordem(
            modelo_tfidf,
            modelo_tfidf.predict_proba(
                X_test_tfidf
            ),
        )
    )

    # ========================================================
    # MODELO B
    # TF-IDF + SENTENCE TRANSFORMER
    # ========================================================

    X_train_hybrid = hstack(
        [
            X_train_tfidf,
            csr_matrix(
                embeddings_train
                * EMBEDDING_WEIGHT
            ),
        ],
        format="csr",
    )

    X_test_hybrid = hstack(
        [
            X_test_tfidf,
            csr_matrix(
                embeddings_test
                * EMBEDDING_WEIGHT
            ),
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
        probabilidades_na_ordem(
            modelo_hybrid,
            modelo_hybrid.predict_proba(
                X_test_hybrid
            ),
        )
    )

    # ========================================================
    # ENSEMBLE FINAL
    # 20% TF-IDF PURO + 80% MODELO HÍBRIDO
    # ========================================================

    proba_final = (
        (
            1.0
            - FINAL_HYBRID_BLEND_WEIGHT
        )
        * proba_tfidf
        +
        FINAL_HYBRID_BLEND_WEIGHT
        * proba_hybrid
    )

    predicoes = np.asarray(
        CLASSES
    )[
        np.argmax(
            proba_final,
            axis=1,
        )
    ]

    # ========================================================
    # ARQUIVO FINAL
    # ========================================================

    saida = pd.DataFrame(
        {
            "resp_text": X_test,
            "clarity": predicoes,
        }
    )

    # ========================================================
    # VALIDAÇÕES
    # ========================================================

    if len(saida) != len(X_test):
        raise RuntimeError(
            "A quantidade de linhas da saída mudou."
        )

    if saida["clarity"].isna().any():
        raise RuntimeError(
            "Existem rótulos vazios na saída."
        )

    rotulos_invalidos = (
        set(saida["clarity"])
        - set(CLASSES)
    )

    if rotulos_invalidos:
        raise RuntimeError(
            "Foram gerados rótulos inválidos: "
            f"{sorted(rotulos_invalidos)}"
        )

    if not (
        saida["resp_text"]
        .reset_index(drop=True)
        .equals(
            X_test.reset_index(
                drop=True
            )
        )
    ):
        raise RuntimeError(
            "A ordem ou o conteúdo dos textos foi alterado."
        )

    saida.to_excel(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nArquivo gerado com sucesso:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nDistribuição das previsões:"
    )

    print(
        saida["clarity"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()

