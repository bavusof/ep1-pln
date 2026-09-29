import pandas as pd

from src.config import (
    ARQUIVO_TREINO,
    COLUNA_TEXTO,
    COLUNA_CLASSE,
)


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def carregar_dados_treino():
    """
    Carrega o conjunto de treinamento e realiza apenas
    as transformações básicas necessárias.

    Importante:
    Não fazemos TF-IDF, normalização ou qualquer outra
    transformação aprendida aqui.

    Essas transformações devem permanecer dentro dos
    Pipelines dos modelos para evitar data leakage.
    """

    df = pd.read_excel(ARQUIVO_TREINO)

    # Respostas ausentes são tratadas como texto vazio.
    # A conversão para string garante compatibilidade
    # com o TfidfVectorizer.
    df[COLUNA_TEXTO] = (
        df[COLUNA_TEXTO]
        .fillna("")
        .astype(str)
    )

    X = df[COLUNA_TEXTO]
    y = df[COLUNA_CLASSE]

    return X, y


def carregar_dataframe_treino():
    """
    Retorna o DataFrame completo.

    Útil para análises exploratórias, distribuição
    das classes e inspeção dos dados.
    """

    df = pd.read_excel(ARQUIVO_TREINO)

    df[COLUNA_TEXTO] = (
        df[COLUNA_TEXTO]
        .fillna("")
        .astype(str)
    )

    return df