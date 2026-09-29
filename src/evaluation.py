import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    RANDOM_STATE,
    N_SPLITS,
    METRICAS,
)


# ============================================================
# VALIDAÇÃO CRUZADA
# ============================================================

def criar_cv():
    """
    Cria a estratégia padrão de validação cruzada do projeto.

    StratifiedKFold é utilizado para preservar, tanto quanto
    possível, a distribuição das classes em cada fold.
    """

    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def avaliar_modelo(modelo, X, y):
    """
    Avalia um Pipeline utilizando a configuração padrão
    de validação cruzada.

    Retorna:
        scores: resultado completo do cross_validate
        resumo: métricas médias e desvios
        resultados_folds: métricas individuais por fold
    """

    cv = criar_cv()

    scores = cross_validate(
        modelo,
        X,
        y,
        cv=cv,
        scoring=METRICAS,
        n_jobs=-1,
        return_train_score=True,
    )

    resultados_folds = pd.DataFrame({
        "fold": range(1, N_SPLITS + 1),

        "accuracy_treino":
            scores["train_accuracy"],

        "accuracy_validacao":
            scores["test_accuracy"],

        "f1_macro_treino":
            scores["train_f1_macro"],

        "f1_macro_validacao":
            scores["test_f1_macro"],
    })

    resumo = {
        "acuracia_media":
            float(scores["test_accuracy"].mean()),

        "desvio_acuracia":
            float(scores["test_accuracy"].std()),

        "f1_macro_medio":
            float(scores["test_f1_macro"].mean()),

        "desvio_f1_macro":
            float(scores["test_f1_macro"].std()),

        "acuracia_treino_media":
            float(scores["train_accuracy"].mean()),

        "f1_macro_treino_medio":
            float(scores["train_f1_macro"].mean()),
    }

    return scores, resumo, resultados_folds