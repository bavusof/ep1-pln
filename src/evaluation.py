from typing import Any, Mapping

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.config import (
    METRICA_SELECAO,
    METRICAS,
    N_SPLITS,
    RANDOM_STATE,
)


def criar_cv(n_splits: int = N_SPLITS) -> StratifiedKFold:
    """Cria a estratégia padrão de validação cruzada estratificada."""
    if n_splits < 2:
        raise ValueError("n_splits deve ser maior ou igual a 2.")

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def avaliar_modelo(
    modelo: Any,
    X: Any,
    y: Any,
    *,
    n_splits: int = N_SPLITS,
):
    """Avalia um pipeline com a CV padrão e retorna scores, resumo e folds."""
    scores = cross_validate(
        modelo,
        X,
        y,
        cv=criar_cv(n_splits),
        scoring=METRICAS,
        n_jobs=-1,
        return_train_score=True,
        error_score="raise",
    )

    resultados_folds = pd.DataFrame(
        {
            "fold": range(1, n_splits + 1),
            "accuracy_treino": scores["train_accuracy"],
            "accuracy_validacao": scores["test_accuracy"],
            "f1_macro_treino": scores["train_f1_macro"],
            "f1_macro_validacao": scores["test_f1_macro"],
        }
    )

    resumo = resumir_scores(scores)
    return scores, resumo, resultados_folds


def resumir_scores(scores: Mapping[str, Any]) -> dict[str, float]:
    """Converte os scores de cross_validate em um resumo padronizado."""
    return {
        "accuracy": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "f1_macro": float(scores["test_f1_macro"].mean()),
        "f1_macro_std": float(scores["test_f1_macro"].std()),
        "train_accuracy": float(scores["train_accuracy"].mean()),
        "train_accuracy_std": float(scores["train_accuracy"].std()),
        "train_f1_macro": float(scores["train_f1_macro"].mean()),
        "train_f1_macro_std": float(scores["train_f1_macro"].std()),
        "accuracy_gap": float(
            scores["train_accuracy"].mean()
            - scores["test_accuracy"].mean()
        ),
    }


def criar_grid_search(
    estimator: Any,
    param_grid: dict | list[dict],
    *,
    n_splits: int = N_SPLITS,
    cv=None,
    refit: str = METRICA_SELECAO,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Cria um GridSearchCV com as convenções do projeto."""
    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv if cv is not None else criar_cv(n_splits),
        scoring=METRICAS,
        refit=refit,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )
