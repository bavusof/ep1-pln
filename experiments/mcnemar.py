"""Teste de McNemar EP1 x EP2 no teste 1 (900 instâncias rotuladas).

Único script que lê o gabarito do teste 1. Rodar uma única vez, depois de
o modelo do EP2 estar definido (a escolha do modelo não usa o teste 1).

    python -m experiments.mcnemar
"""

import pandas as pd
from scipy.stats import binomtest, chi2

from src.config import (
    ARQUIVO_EP1_TESTE1,
    ARQUIVO_TESTE1_EP2,
    ARQUIVO_TESTE1_ROTULADO,
    COLUNA_CLASSE,
    COLUNA_TEXTO,
    EP2_RESULTS_DIR,
)
from src.results import salvar_tabela


def main() -> None:
    gabarito = pd.read_excel(ARQUIVO_TESTE1_ROTULADO)
    ep1 = pd.read_excel(ARQUIVO_EP1_TESTE1)
    ep2 = pd.read_excel(ARQUIVO_TESTE1_EP2)
    for nome, df in (("EP1", ep1), ("EP2", ep2)):
        assert len(df) == len(gabarito), f"{nome}: {len(df)} linhas"
        assert (df[COLUNA_TEXTO].fillna("") == gabarito[COLUNA_TEXTO].fillna("")).all(), f"{nome}: textos fora de ordem"

    acerto_ep1 = (ep1[COLUNA_CLASSE] == gabarito[COLUNA_CLASSE]).to_numpy()
    acerto_ep2 = (ep2[COLUNA_CLASSE] == gabarito[COLUNA_CLASSE]).to_numpy()
    a = int((acerto_ep1 & acerto_ep2).sum())    # ambos acertam
    b = int((acerto_ep1 & ~acerto_ep2).sum())   # só EP1 acerta
    c = int((~acerto_ep1 & acerto_ep2).sum())   # só EP2 acerta
    d = int((~acerto_ep1 & ~acerto_ep2).sum())  # ambos erram

    p_exato = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    estatistica = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    p_chi2 = float(chi2.sf(estatistica, df=1))

    resultado = pd.DataFrame(
        [
            {
                "acc_ep1": acerto_ep1.mean(),
                "acc_ep2": acerto_ep2.mean(),
                "ambos_acertam": a,
                "so_ep1_acerta": b,
                "so_ep2_acerta": c,
                "ambos_erram": d,
                "mcnemar_exato_p": p_exato,
                "mcnemar_chi2_corrigido": estatistica,
                "mcnemar_chi2_p": p_chi2,
                "significativo_5pct": bool(p_exato < 0.05 and c > b),
            }
        ]
    )
    salvar_tabela(resultado, EP2_RESULTS_DIR / "mcnemar.csv")
    print(resultado.T.to_string(header=False))


if __name__ == "__main__":
    main()
