from typing import Any, Mapping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.embeddings import TfidfWeightedWord2VecDocumentTransformer, Word2VecDocumentTransformer

from src.features import TextStructuralFeatures

from src.config import RANDOM_STATE


DEFAULT_LOGISTIC_PARAMS = {
    "max_iter": 3000,
    "random_state": RANDOM_STATE,
}


def criar_regressao_logistica(C: float = 1.0) -> LogisticRegression:
    """Cria a Regressão Logística padrão do projeto."""
    return LogisticRegression(
        C=C,
        **DEFAULT_LOGISTIC_PARAMS,
    )


def criar_tfidf_word(
    *,
    ngram_range=(1, 1),
    min_df=1,
    max_df=1.0,
    sublinear_tf=False,
    **kwargs: Any,
) -> TfidfVectorizer:
    """Cria um TF-IDF baseado em n-gramas de palavras."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        **kwargs,
    )


def criar_tfidf_caracteres(
    *,
    analyzer: str = "char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_df=1.0,
    sublinear_tf=True,
    **kwargs: Any,
) -> TfidfVectorizer:
    """Cria um TF-IDF baseado em n-gramas de caracteres."""
    if analyzer not in {"char", "char_wb"}:
        raise ValueError("analyzer deve ser 'char' ou 'char_wb'.")

    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        **kwargs,
    )

def criar_features_word_structural(
    *,
    ngram_range=(1, 2),
    min_df=5,
    max_df=1.0,
    sublinear_tf=True,
    structural_weight=1.0,
):
    """
    Combina TF-IDF de palavras com características estruturais.
    """

    return FeatureUnion(
        [
            (
                "tfidf",
                criar_tfidf_word(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=sublinear_tf,
                ),
            ),
            (
                "structural",
                Pipeline(
                    [
                        (
                            "extract",
                            TextStructuralFeatures(),
                        ),
                        (
                            "scale",
                            StandardScaler(
                                with_mean=False,
                            ),
                        ),
                    ]
                ),
            ),
        ],
        transformer_weights={
            "structural": structural_weight,
        },
    )

def criar_pipeline_tfidf_word_structural(
    *,
    ngram_range=(1, 2),
    min_df=5,
    max_df=1.0,
    sublinear_tf=True,
    C=0.5,
    structural_weight=1.0,
) -> Pipeline:
    """TF-IDF de palavras + features estruturais + LR."""

    features = criar_features_word_structural(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        structural_weight=structural_weight,
    )

    return Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "logistic",
                criar_regressao_logistica(
                    C=C
                ),
            ),
        ]
    )

def criar_pipeline_tfidf_word_structural_svc(
    *,
    ngram_range=(1, 2),
    min_df=5,
    max_df=1.0,
    sublinear_tf=True,
    C=0.25,
    structural_weight=1.0,
) -> Pipeline:
    """TF-IDF de palavras + features estruturais + LinearSVC."""

    features = criar_features_word_structural(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        structural_weight=structural_weight,
    )

    return Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "svc",
                LinearSVC(
                    C=C,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

def criar_pipeline_tfidf_word(
    *,
    ngram_range=(1, 1),
    min_df=1,
    max_df=1.0,
    sublinear_tf=False,
    C=1.0,
    preprocessor=None,
    stop_words=None,
    strip_accents=None,
) -> Pipeline:
    """Cria TF-IDF de palavras + Regressão Logística."""

    return Pipeline(
        [
            (
                "tfidf",
                criar_tfidf_word(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=sublinear_tf,
                    preprocessor=preprocessor,
                    stop_words=stop_words,
                    strip_accents=strip_accents,
                ),
            ),
            (
                "logistic",
                criar_regressao_logistica(C=C)
            ),
        ]
    )


def criar_pipeline_tfidf_caracteres(
    *,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_df=1.0,
    sublinear_tf=True,
    C=1.0,
) -> Pipeline:
    """Cria TF-IDF de caracteres + Regressão Logística."""
    return Pipeline(
        [
            (
                "tfidf",
                criar_tfidf_caracteres(
                    analyzer=analyzer,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=sublinear_tf,
                ),
            ),
            ("logistic", criar_regressao_logistica(C=C)),
        ]
    )

def criar_pipeline_tfidf_word_char(
    *,
    word_ngram_range=(1, 2),
    word_min_df=2,
    char_ngram_range=(3, 5),
    char_min_df=2,
    C=1.0,
) -> Pipeline:
    """Cria a combinação TF-IDF de palavras + caracteres."""
    features = FeatureUnion(
        [
            (
                "word",
                criar_tfidf_word(
                    ngram_range=word_ngram_range,
                    min_df=word_min_df,
                ),
            ),
            (
                "char",
                criar_tfidf_caracteres(
                    analyzer="char_wb",
                    ngram_range=char_ngram_range,
                    min_df=char_min_df,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            ("logistic", criar_regressao_logistica(C=C)),
        ]
    )

def criar_pipeline_tfidf_word_selecao(
    *,
    ngram_range=(1, 2),
    min_df=5,
    max_df=1.0,
    sublinear_tf=True,
    C=0.5,
    percentile=100,
    preprocessor=None,
    stop_words=None,
    strip_accents=None,
) -> Pipeline:
    """
    Cria TF-IDF de palavras + seleção de atributos por
    qui-quadrado + Regressão Logística.
    """

    return Pipeline(
        [
            (
                "tfidf",
                criar_tfidf_word(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=sublinear_tf,
                    preprocessor=preprocessor,
                    stop_words=stop_words,
                    strip_accents=strip_accents,
                ),
            ),
            (
                "selection",
                SelectPercentile(
                    score_func=chi2,
                    percentile=percentile,
                ),
            ),
            (
                "logistic",
                criar_regressao_logistica(C=C),
            ),
        ]
    )

def criar_pipeline_word2vec(
    *,
    vector_size=200,
    window=5,
    min_count=2,
    epochs=5,
    sg=1,
    C=0.5,
) -> Pipeline:
    """Word2Vec documental + Regressão Logística."""

    return Pipeline(
        [
            (
                "embedding",
                Word2VecDocumentTransformer(
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    epochs=epochs,
                    sg=sg,
                    workers=1,
                    seed=RANDOM_STATE,
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "logistic",
                criar_regressao_logistica(C=C),
            ),
        ]
    )

def criar_pipeline_tfidf_weighted_word2vec(
    *,
    vector_size=200,
    window=5,
    min_count=2,
    epochs=5,
    sg=1,
    C=0.5,
    tfidf_ngram_range=(1, 1),
    tfidf_min_df=5,
    tfidf_max_df=1.0,
    tfidf_sublinear_tf=True,
    tfidf_preprocessor=None,
    tfidf_stop_words=None,
    tfidf_strip_accents=None,
) -> Pipeline:
    """
    TF-IDF-weighted Word2Vec + Regressão Logística.

    O TF-IDF fornece a importância das palavras e o Word2Vec
    fornece a representação semântica.
    """

    return Pipeline(
        [
            (
                "embedding",
                TfidfWeightedWord2VecDocumentTransformer(
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    epochs=epochs,
                    sg=sg,
                    workers=1,
                    seed=RANDOM_STATE,
                    tfidf_ngram_range=tfidf_ngram_range,
                    tfidf_min_df=tfidf_min_df,
                    tfidf_max_df=tfidf_max_df,
                    tfidf_sublinear_tf=tfidf_sublinear_tf,
                    tfidf_preprocessor=tfidf_preprocessor,
                    tfidf_stop_words=tfidf_stop_words,
                    tfidf_strip_accents=tfidf_strip_accents,
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "logistic",
                criar_regressao_logistica(C=C),
            ),
        ]
    )

def criar_pipeline_hybrid_tfidf_word2vec(
    *,
    word_ngram_range=(1, 2),
    word_min_df=5,
    word_max_df=1.0,
    word_sublinear_tf=True,
    vector_size=200,
    window=5,
    min_count=2,
    epochs=5,
    sg=1,
    embedding_tfidf_ngram_range=(1, 1),
    embedding_tfidf_min_df=5,
    embedding_tfidf_max_df=1.0,
    embedding_tfidf_sublinear_tf=True,
    embedding_weight=0.5,
    C=0.5,
) -> Pipeline:
    """
    Híbrido:

        TF-IDF de palavras
                +
        TF-IDF-weighted Word2Vec

    seguido de Regressão Logística.
    """

    features = FeatureUnion(
        [
            (
                "tfidf",
                criar_tfidf_word(
                    ngram_range=word_ngram_range,
                    min_df=word_min_df,
                    max_df=word_max_df,
                    sublinear_tf=word_sublinear_tf,
                ),
            ),
            (
                "embedding",
                Pipeline(
                    [
                        (
                            "weighted_word2vec",
                            TfidfWeightedWord2VecDocumentTransformer(
                                vector_size=vector_size,
                                window=window,
                                min_count=min_count,
                                epochs=epochs,
                                sg=sg,
                                workers=1,
                                seed=RANDOM_STATE,
                                tfidf_ngram_range=(
                                    embedding_tfidf_ngram_range
                                ),
                                tfidf_min_df=(
                                    embedding_tfidf_min_df
                                ),
                                tfidf_max_df=(
                                    embedding_tfidf_max_df
                                ),
                                tfidf_sublinear_tf=(
                                    embedding_tfidf_sublinear_tf
                                ),
                            ),
                        ),
                        (
                            "scale",
                            StandardScaler(),
                        ),
                    ]
                ),
            ),
        ],
        transformer_weights={
            "embedding": embedding_weight,
        },
    )

    return Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "logistic",
                criar_regressao_logistica(C=C),
            ),
        ]
    )

def criar_pipeline(config: Mapping[str, Any]) -> Pipeline:
    """Cria um pipeline a partir de uma configuração de experimento."""
    tipo = config["tipo"]
    parametros = config.get("parametros", {})

    if tipo == "tfidf_word":
        return criar_pipeline_tfidf_word(**parametros)

    if tipo == "tfidf_char":
        return criar_pipeline_tfidf_caracteres(**parametros)

    if tipo == "tfidf_word_char":
        return criar_pipeline_tfidf_word_char(**parametros)

    if tipo == "tfidf_word_selecao":
        return criar_pipeline_tfidf_word_selecao(**parametros)

    if tipo == "tfidf_word_structural":
        return criar_pipeline_tfidf_word_structural(**parametros)

    if tipo == "tfidf_word_structural_svc":
        return criar_pipeline_tfidf_word_structural_svc(**parametros)

    if tipo == "word2vec":
        return criar_pipeline_word2vec(**parametros)

    if tipo == "tfidf_weighted_word2vec":
        return criar_pipeline_tfidf_weighted_word2vec(**parametros)

    if tipo == "hybrid_tfidf_word2vec":
        return criar_pipeline_hybrid_tfidf_word2vec(**parametros)

    raise ValueError(f"Tipo de pipeline desconhecido: {tipo}")


def criar_baseline_majoritario() -> Any:
    """Cria o baseline que sempre prevê a classe mais frequente."""
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="most_frequent")
