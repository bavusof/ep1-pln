"""Treino limpo do EP2: tratamento aplicado só à parte de treino de cada fold.

1. Duplicatas exatas (após limpar_para_bert): uma linha por texto, com o
   rótulo majoritário; grupos empatados entre classes são descartados.
2. Textos sem conteúdo (<= MIN_PALAVRAS palavras), cujos rótulos são ~uniformes.

A validação de cada fold não passa por aqui: continua com textos e rótulos
originais, para a comparação com as bases atuais ser justa.
"""

import numpy as np
import pandas as pd

from src.text_preprocessing import limpar_para_bert

MIN_PALAVRAS = 5


def tratar_treino(textos, rotulos) -> tuple[np.ndarray, np.ndarray, dict]:
    """Retorna (textos, rótulos, relatório). Os textos devolvidos são os originais
    (não limpos), para cada modelo aplicar o próprio pré-processamento."""
    df = pd.DataFrame({"texto": np.asarray(textos, dtype=object), "rotulo": np.asarray(rotulos)})
    df["chave"] = df["texto"].map(limpar_para_bert)
    n_inicial = len(df)

    curto = df["chave"].str.split().str.len().fillna(0) <= MIN_PALAVRAS
    df = df[~curto]

    contagem = df.groupby(["chave", "rotulo"]).size().unstack(fill_value=0)
    maximo = contagem.max(axis=1)
    empatado = (contagem.eq(maximo, axis=0).sum(axis=1) > 1)
    majoritario = contagem.idxmax(axis=1)
    repetidos = contagem.sum(axis=1) > 1
    conflitantes = (contagem > 0).sum(axis=1) > 1

    validos = majoritario[~empatado]
    primeira = df.drop_duplicates("chave").set_index("chave")
    primeira = primeira.loc[primeira.index.isin(validos.index)]
    rotulo_final = validos.loc[primeira.index]

    relatorio = {
        "linhas_iniciais": n_inicial,
        "removidas_curtas": int(curto.sum()),
        "textos_repetidos": int(repetidos.sum()),
        "textos_conflitantes": int(conflitantes.sum()),
        "grupos_empatados_descartados": int(empatado.sum()),
        "linhas_finais": int(len(primeira)),
    }
    return primeira["texto"].to_numpy(dtype=object), rotulo_final.to_numpy(), relatorio
