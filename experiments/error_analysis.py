from pathlib import Path

from experiments.current import CURRENT_MODEL

from src.data import carregar_dados_treino
from src.error_analysis import preparar_analise_erros


RESULTS_DIR = Path(
    "results/error_analysis"
)


def main() -> None:
    X, y = carregar_dados_treino()

    analise = preparar_analise_erros(
        CURRENT_MODEL,
        X,
        y,
    )

    resultados = analise["resultados"]
    resumo = analise["resumo"]
    matriz = analise["matriz"]
    relatorio = analise["relatorio"]
    confusoes = analise["confusoes"]

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # RESUMO
    # ========================================================

    print("\n==============================")
    print("ANÁLISE DE ERROS")
    print("==============================")

    print(
        f"Modelo: {CURRENT_MODEL['name']}"
    )

    print(
        f"Quantidade de exemplos: "
        f"{resumo['quantidade']}"
    )

    print(
        f"Acurácia: "
        f"{resumo['accuracy']:.4f}"
    )

    print(
        f"Quantidade de erros: "
        f"{resumo['quantidade_erros']}"
    )

    print(
        f"Taxa de erro: "
        f"{resumo['taxa_erro']:.4f}"
    )

    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    print("\n==============================")
    print("MATRIZ DE CONFUSÃO")
    print("==============================")

    print(matriz)

    # ========================================================
    # RELATÓRIO
    # ========================================================

    print("\n==============================")
    print("CLASSIFICATION REPORT")
    print("==============================")

    print(relatorio)

    # ========================================================
    # PRINCIPAIS CONFUSÕES
    # ========================================================

    print("\n==============================")
    print("PRINCIPAIS CONFUSÕES")
    print("==============================")

    print(
        confusoes.to_string(
            index=False
        )
    )

    # ========================================================
    # SALVAMENTO
    # ========================================================

    resultados.to_csv(
        RESULTS_DIR / "predicoes_holdout.csv",
        index=False,
    )

    matriz.to_csv(
        RESULTS_DIR / "matriz_confusao.csv"
    )

    relatorio.to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    confusoes.to_csv(
        RESULTS_DIR / "resumo_confusoes.csv",
        index=False,
    )

    # Apenas os erros.
    resultados[
        ~resultados["acertou"]
    ].to_csv(
        RESULTS_DIR / "erros.csv",
        index=False,
    )

    # Erros nos quais o modelo estava mais confiante.
    (
        resultados[
            ~resultados["acertou"]
        ]
        .sort_values(
            "confianca",
            ascending=False,
        )
        .head(100)
        .to_csv(
            RESULTS_DIR
            / "erros_confianca_alta.csv",
            index=False,
        )
    )

    # Erros nos quais a decisão foi menos confiante.
    (
        resultados[
            ~resultados["acertou"]
        ]
        .sort_values(
            "confianca",
            ascending=True,
        )
        .head(100)
        .to_csv(
            RESULTS_DIR
            / "erros_confianca_baixa.csv",
            index=False,
        )
    )

    print(
        "\nArquivos salvos em:",
        RESULTS_DIR,
    )


if __name__ == "__main__":
    main()