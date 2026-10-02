from typing import Any, Mapping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_selection import SelectPercentile, chi2

from src.config import RANDOM_STATE


DEFAULT_LOGISTIC_PARAMS = {
    "max_iter": 1000,
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

    raise ValueError(f"Tipo de pipeline desconhecido: {tipo}")


def criar_baseline_majoritario() -> Any:
    """Cria o baseline que sempre prevê a classe mais frequente."""
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="most_frequent")

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
