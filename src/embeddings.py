from typing import Any, Iterable

import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted


class Word2VecDocumentTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Treina Word2Vec no conjunto recebido e representa cada documento
    pela média dos embeddings das palavras conhecidas.
    """

    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 5,
        sg: int = 1,
        workers: int = 1,
        seed: int = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.workers = workers
        self.seed = seed

    @staticmethod
    def _normalizar_textos(X: Iterable[Any]) -> list[str]:
        return [
            "" if value is None else str(value)
            for value in X
        ]

    def fit(self, X, y=None):
        textos = self._normalizar_textos(X)

        sentencas = [
            texto.split()
            for texto in textos
            if texto.split()
        ]

        if not sentencas:
            raise ValueError(
                "Nenhum texto válido foi encontrado para treinar Word2Vec."
            )

        self.word2vec_ = Word2Vec(
            sentences=sentencas,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=self.sg,
            workers=self.workers,
            seed=self.seed,
        )

        return self

    def transform(self, X):
        check_is_fitted(self, "word2vec_")

        textos = self._normalizar_textos(X)

        vectors = np.zeros(
            (len(textos), self.vector_size),
            dtype=np.float32,
        )

        for i, texto in enumerate(textos):
            palavras = texto.split()

            validas = [
                palavra
                for palavra in palavras
                if palavra in self.word2vec_.wv.key_to_index
            ]

            if validas:
                vectors[i] = np.mean(
                    [
                        self.word2vec_.wv[word]
                        for word in validas
                    ],
                    axis=0,
                )

        return vectors


class TfidfWeightedWord2VecDocumentTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Representa cada documento pela média ponderada dos embeddings Word2Vec,
    utilizando os pesos TF-IDF das palavras.

    O TF-IDF e o Word2Vec são ajustados exclusivamente sobre X recebido
    em fit(), portanto a transformação é compatível com validação cruzada
    sem vazamento entre treino e validação.
    """

    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 5,
        sg: int = 1,
        workers: int = 1,
        seed: int = 42,
        tfidf_ngram_range=(1, 1),
        tfidf_min_df: int = 5,
        tfidf_max_df: float = 1.0,
        tfidf_sublinear_tf: bool = True,
        tfidf_preprocessor=None,
        tfidf_stop_words=None,
        tfidf_strip_accents=None,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.workers = workers
        self.seed = seed

        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_sublinear_tf = tfidf_sublinear_tf
        self.tfidf_preprocessor = tfidf_preprocessor
        self.tfidf_stop_words = tfidf_stop_words
        self.tfidf_strip_accents = tfidf_strip_accents

    @staticmethod
    def _normalizar_textos(X: Iterable[Any]) -> list[str]:
        return [
            "" if value is None else str(value)
            for value in X
        ]

    def fit(self, X, y=None):
        textos = self._normalizar_textos(X)

        # ----------------------------------------------------
        # TF-IDF usado para calcular os pesos
        # ----------------------------------------------------

        self.tfidf_ = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.tfidf_ngram_range,
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            sublinear_tf=self.tfidf_sublinear_tf,
            preprocessor=self.tfidf_preprocessor,
            stop_words=self.tfidf_stop_words,
            strip_accents=self.tfidf_strip_accents,
        )

        self.tfidf_.fit(textos)

        # O mesmo analisador do TF-IDF será usado para construir
        # as sentenças destinadas ao Word2Vec.
        self.analyzer_ = self.tfidf_.build_analyzer()

        sentencas = []

        for texto in textos:
            tokens = self.analyzer_(texto)

            if tokens:
                sentencas.append(tokens)

        if not sentencas:
            raise ValueError(
                "Nenhum token válido foi encontrado "
                "para treinar Word2Vec."
            )

        # ----------------------------------------------------
        # Word2Vec
        # ----------------------------------------------------

        self.word2vec_ = Word2Vec(
            sentences=sentencas,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=self.sg,
            workers=self.workers,
            seed=self.seed,
        )

        # ----------------------------------------------------
        # Mapeamento:
        # coluna TF-IDF -> índice do vetor Word2Vec
        # ----------------------------------------------------

        feature_names = self.tfidf_.get_feature_names_out()

        self.feature_to_w2v_index_ = np.full(
            len(feature_names),
            -1,
            dtype=np.int32,
        )

        for column, token in enumerate(feature_names):
            index = self.word2vec_.wv.key_to_index.get(token)

            if index is not None:
                self.feature_to_w2v_index_[column] = index

        return self

    def transform(self, X):
        check_is_fitted(
            self,
            [
                "tfidf_",
                "word2vec_",
                "feature_to_w2v_index_",
            ],
        )

        textos = self._normalizar_textos(X)

        matriz_tfidf = (
            self.tfidf_
            .transform(textos)
            .tocsr()
        )

        vectors = np.zeros(
            (len(textos), self.vector_size),
            dtype=np.float32,
        )

        matriz_word_vectors = (
            self.word2vec_.wv.vectors
        )

        for i in range(matriz_tfidf.shape[0]):

            inicio = matriz_tfidf.indptr[i]
            fim = matriz_tfidf.indptr[i + 1]

            indices = matriz_tfidf.indices[
                inicio:fim
            ]

            pesos = matriz_tfidf.data[
                inicio:fim
            ]

            total_peso = 0.0
            acumulado = vectors[i]

            for coluna, peso in zip(indices, pesos):

                indice_w2v = (
                    self.feature_to_w2v_index_[coluna]
                )

                if indice_w2v < 0:
                    continue

                acumulado += (
                    peso
                    * matriz_word_vectors[indice_w2v]
                )

                total_peso += float(peso)

            # Média ponderada.
            if total_peso > 0.0:
                acumulado /= total_peso

        return vectors