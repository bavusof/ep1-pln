from typing import Any

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSES,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.models import criar_pipeline


def criar_holdout(X: Any, y: Any):
    """
    Cria a divisão estratificada utilizada pela análise de erros.

    Esta divisão é utilizada somente para inspeção dos erros.
    Ela não substitui a validação cruzada utilizada para avaliar
    e comparar modelos.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def obter_predicoes_holdout(
    modelo: Any,
    X: Any,
    y: Any,
) -> pd.DataFrame:
    """
    Treina o modelo em um holdout e retorna uma tabela contendo:

    - texto;
    - classe real;
    - classe predita;
    - probabilidades;
    - confiança da previsão;
    - tamanho do texto.
    """

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    ) = criar_holdout(X, y)

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_holdout)
    probabilidades = modelo.predict_proba(X_holdout)

    resultados = pd.DataFrame(
        {
            "texto": X_holdout.reset_index(drop=True),
            "classe_real": y_holdout.reset_index(drop=True),
            "classe_predita": y_pred,
        }
    )

    # Probabilidade atribuída a cada classe.
    for indice, classe in enumerate(modelo.classes_):
        resultados[f"prob_{classe}"] = probabilidades[:, indice]

    resultados["confianca"] = probabilidades.max(axis=1)

    # Características simples do texto.
    resultados["n_caracteres"] = (
        resultados["texto"].str.len()
    )

    resultados["n_palavras"] = (
        resultados["texto"]
        .str.split()
        .str.len()
    )

    resultados["acertou"] = (
        resultados["classe_real"]
        == resultados["classe_predita"]
    )

    return resultados


def calcular_resumo_erros(
    resultados: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calcula métricas gerais da análise de erros.
    """

    y_real = resultados["classe_real"]
    y_pred = resultados["classe_predita"]

    return {
        "accuracy": accuracy_score(
            y_real,
            y_pred,
        ),
        "quantidade": len(resultados),
        "quantidade_erros": int(
            (~resultados["acertou"]).sum()
        ),
        "taxa_erro": float(
            (~resultados["acertou"]).mean()
        ),
    }


def criar_matriz_confusao(
    resultados: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria a matriz de confusão respeitando a ordem oficial
    das classes.
    """

    matriz = confusion_matrix(
        resultados["classe_real"],
        resultados["classe_predita"],
        labels=CLASSES,
    )

    return pd.DataFrame(
        matriz,
        index=[
            f"Real {classe}"
            for classe in CLASSES
        ],
        columns=[
            f"Predito {classe}"
            for classe in CLASSES
        ],
    )


def criar_classification_report(
    resultados: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria o relatório de classificação por classe.
    """

    relatorio = classification_report(
        resultados["classe_real"],
        resultados["classe_predita"],
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(relatorio).transpose()


def criar_resumo_por_confusao(
    resultados: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resume as principais confusões entre classe real
    e classe predita.
    """

    erros = resultados[
        ~resultados["acertou"]
    ].copy()

    resumo = (
        erros
        .groupby(
            [
                "classe_real",
                "classe_predita",
            ]
        )
        .size()
        .reset_index(
            name="quantidade"
        )
        .sort_values(
            "quantidade",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return resumo


def preparar_analise_erros(
    current_model_config: dict,
    X: Any,
    y: Any,
):
    """
    Executa toda a análise de erros usando exatamente
    a configuração definida como current_model.
    """

    pipeline = criar_pipeline(
        {
            "tipo": current_model_config["type"],
            "parametros": current_model_config[
                "parameters"
            ],
        }
    )

    resultados = obter_predicoes_holdout(
        pipeline,
        X,
        y,
    )

    resumo = calcular_resumo_erros(
        resultados
    )

    matriz = criar_matriz_confusao(
        resultados
    )

    relatorio = criar_classification_report(
        resultados
    )

    resumo_confusoes = (
        criar_resumo_por_confusao(
            resultados
        )
    )

    return {
        "resultados": resultados,
        "resumo": resumo,
        "matriz": matriz,
        "relatorio": relatorio,
        "confusoes": resumo_confusoes,
    }