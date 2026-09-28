from pathlib import Path

import pandas as pd


ARQUIVO_RESULTADOS = Path("resultados_experimentos.csv")


def registrar_resultado(resultado: dict) -> None:
    """Registra/atualiza um experimento na tabela consolidada."""
    novo = pd.DataFrame([resultado])

    if ARQUIVO_RESULTADOS.exists():
        atual = pd.read_csv(ARQUIVO_RESULTADOS)
        experimento = resultado.get("experimento")
        if experimento is not None and "experimento" in atual.columns:
            atual = atual[atual["experimento"] != experimento]
        df_resultados = pd.concat([atual, novo], ignore_index=True)
    else:
        df_resultados = novo

    df_resultados.to_csv(ARQUIVO_RESULTADOS, index=False)
    print(f"Tabela consolidada salva em: {ARQUIVO_RESULTADOS}")
