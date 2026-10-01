import pandas as pd

from src.config import ARQUIVO_TREINO, COLUNA_CLASSE, COLUNA_TEXTO


def carregar_dataframe_treino() -> pd.DataFrame:
    """Carrega o DataFrame completo e aplica apenas tratamentos básicos."""
    if not ARQUIVO_TREINO.exists():
        raise FileNotFoundError(
            f"Arquivo de treino não encontrado: {ARQUIVO_TREINO}\n"
            "Coloque train.xlsx dentro da pasta data/."
        )

    df = pd.read_excel(ARQUIVO_TREINO)

    colunas_obrigatorias = {COLUNA_TEXTO, COLUNA_CLASSE}
    colunas_ausentes = colunas_obrigatorias - set(df.columns)
    if colunas_ausentes:
        raise ValueError(
            "O arquivo de treino não possui as colunas obrigatórias: "
            f"{sorted(colunas_ausentes)}"
        )

    # Respostas ausentes são tratadas como texto vazio.
    # A conversão para string garante compatibilidade com os vetorizadores.
    df[COLUNA_TEXTO] = df[COLUNA_TEXTO].fillna("").astype(str)

    return df


def carregar_dados_treino():
    """Retorna X (textos) e y (classes)."""
    df = carregar_dataframe_treino()
    return df[COLUNA_TEXTO], df[COLUNA_CLASSE]
