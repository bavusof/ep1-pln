import re

import numpy as np

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


class TextStructuralFeatures(
    BaseEstimator,
    TransformerMixin,
):
    """
    Extrai características estruturais simples do texto.

    Essas características não tentam representar o conteúdo
    lexical; elas descrevem propriedades formais da resposta.
    """

    FEATURE_NAMES = [
        "n_caracteres",
        "n_palavras",
        "n_frases",
        "media_caracteres_palavra",
        "media_palavras_frase",
        "riqueza_lexical",
        "n_digitos",
        "n_numeros",
        "n_urls",
        "n_exclamacoes",
        "n_interrogacoes",
        "n_virgulas",
        "n_ponto_virgulas",
        "proporcao_maiusculas",
    ]

    URL_PATTERN = re.compile(
        r"(https?://|www\.)\S+",
        flags=re.IGNORECASE,
    )

    NUMBER_PATTERN = re.compile(
        r"\b\d+(?:[./-]\d+)*\b"
    )

    WORD_PATTERN = re.compile(
        r"\b[\wÀ-ÿ]+\b",
        flags=re.UNICODE,
    )

    SENTENCE_PATTERN = re.compile(
        r"[.!?]+"
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        linhas = []

        for texto in X:
            texto = "" if texto is None else str(texto)

            palavras = self.WORD_PATTERN.findall(
                texto
            )

            n_palavras = len(palavras)

            n_caracteres = len(texto)

            frases = self.SENTENCE_PATTERN.findall(
                texto
            )

            n_frases = len(frases)

            n_digitos = sum(
                caractere.isdigit()
                for caractere in texto
            )

            n_numeros = len(
                self.NUMBER_PATTERN.findall(
                    texto
                )
            )

            n_urls = len(
                self.URL_PATTERN.findall(
                    texto
                )
            )

            n_exclamacoes = texto.count("!")
            n_interrogacoes = texto.count("?")
            n_virgulas = texto.count(",")
            n_ponto_virgulas = texto.count(";")

            letras = [
                caractere
                for caractere in texto
                if caractere.isalpha()
            ]

            n_letras = len(letras)

            if n_letras > 0:
                proporcao_maiusculas = (
                    sum(
                        caractere.isupper()
                        for caractere in letras
                    )
                    / n_letras
                )
            else:
                proporcao_maiusculas = 0.0

            if n_palavras > 0:
                media_caracteres_palavra = (
                    sum(
                        len(palavra)
                        for palavra in palavras
                    )
                    / n_palavras
                )

                palavras_unicas = {
                    palavra.lower()
                    for palavra in palavras
                }

                riqueza_lexical = (
                    len(palavras_unicas)
                    / n_palavras
                )
            else:
                media_caracteres_palavra = 0.0
                riqueza_lexical = 0.0

            if n_frases > 0:
                media_palavras_frase = (
                    n_palavras
                    / n_frases
                )
            else:
                media_palavras_frase = (
                    float(n_palavras)
                )

            linhas.append(
                [
                    n_caracteres,
                    n_palavras,
                    n_frases,
                    media_caracteres_palavra,
                    media_palavras_frase,
                    riqueza_lexical,
                    n_digitos,
                    n_numeros,
                    n_urls,
                    n_exclamacoes,
                    n_interrogacoes,
                    n_virgulas,
                    n_ponto_virgulas,
                    proporcao_maiusculas,
                ]
            )

        return csr_matrix(
            np.asarray(
                linhas,
                dtype=np.float64,
            )
        )

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return np.asarray(
            self.FEATURE_NAMES,
            dtype=object,
        )