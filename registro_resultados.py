from pathlib import Path

import pandas as pd


ARQUIVO_RESULTADOS = Path("results/resultados_experimentos.csv")


# ============================================================
# REGISTRO DE EXPERIMENTOS
# ============================================================

def registrar_resultado(resultado: dict) -> None:
    """
    Adiciona um experimento à tabela consolidada.

    Se já existir um experimento com o mesmo nome,
    ele é substituído.

    Isso permite executar novamente um experimento sem
    acumular duplicatas no arquivo de resultados.
    """

    novo = pd.DataFrame([resultado])

    if ARQUIVO_RESULTADOS.exists():

        atual = pd.read_csv(ARQUIVO_RESULTADOS)

        nome_experimento = resultado.get("experimento")

        if (
            nome_experimento is not None
            and "experimento" in atual.columns
        ):
            atual = atual[
                atual["experimento"] != nome_experimento
            ]

        df_resultados = pd.concat(
            [atual, novo],
            ignore_index=True,
        )

    else:
        df_resultados = novo

    df_resultados.to_csv(
        ARQUIVO_RESULTADOS,
        index=False,
    )

    print(
        f"Tabela consolidada salva em: "
        f"{ARQUIVO_RESULTADOS}"
    )