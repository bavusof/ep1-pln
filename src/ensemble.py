"""Componentes compartilhados pelo ensemble e pela predição final."""

from itertools import product
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import CLASSES
from src.models import criar_pipeline_tfidf_caracteres, criar_pipeline_tfidf_word


# Melhores configurações clássicas encontradas no tuning (results/tuning/).
MODELOS_CLASSICOS = {
    "tfidf_word": lambda: criar_pipeline_tfidf_word(
        ngram_range=(1, 2), min_df=5, max_df=1.0, sublinear_tf=True, C=0.5
    ),
    "tfidf_char": lambda: criar_pipeline_tfidf_caracteres(
        analyzer="char", ngram_range=(4, 6), min_df=2, sublinear_tf=True, C=0.5
    ),
}


def probabilidades_classicos(X_train, y_train, X_pred) -> dict[str, np.ndarray]:
    """Treina os modelos clássicos e retorna probabilidades na ordem de CLASSES."""
    saida = {}
    for nome, fabrica in MODELOS_CLASSICOS.items():
        modelo = fabrica().fit(X_train, y_train)
        ordem = [list(modelo.classes_).index(c) for c in CLASSES]
        saida[nome] = modelo.predict_proba(X_pred)[:, ordem]
    return saida


def combinar(probs: Mapping[str, np.ndarray], pesos: Mapping[str, float]) -> np.ndarray:
    return sum(pesos[nome] * probs[nome] for nome in pesos)


def grade_pesos(nomes: Sequence[str], passo: float = 0.05):
    """Gera combinações de pesos não negativos que somam 1."""
    n = round(1 / passo)
    for partes in product(range(n + 1), repeat=len(nomes) - 1):
        if sum(partes) <= n:
            yield dict(zip(nomes, [p / n for p in partes] + [(n - sum(partes)) / n]))


def melhores_pesos(probs, y_true, passo: float = 0.05) -> tuple[dict, float]:
    y_true = np.asarray(y_true)
    melhor, melhor_acc = None, -1.0
    for pesos in grade_pesos(list(probs), passo):
        pred = np.array(CLASSES)[combinar(probs, pesos).argmax(1)]
        acc = float((pred == y_true).mean())
        if acc > melhor_acc + 1e-12:
            melhor, melhor_acc = pesos, acc
    return melhor, melhor_acc


def tabela_lookup(X_train, y_train, pureza_minima: float = 1.0) -> dict[str, str]:
    """Mapeia textos repetidos no treino para o rótulo dominante.

    Só entram textos cuja classe dominante tem fração >= pureza_minima
    (1.0 = apenas textos sem rótulos conflitantes).
    """
    df = pd.DataFrame({"texto": np.asarray(X_train), "classe": np.asarray(y_train)})
    contagem = df.groupby(["texto", "classe"]).size().unstack(fill_value=0)
    fracao = contagem.max(axis=1) / contagem.sum(axis=1)
    validos = contagem[fracao >= pureza_minima]
    return validos.idxmax(axis=1).to_dict()


def aplicar_lookup(textos, pred, tabela: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred).copy()
    mascara = np.array([t in tabela for t in textos])
    pred[mascara] = [tabela[t] for t in np.asarray(textos)[mascara]]
    return pred, mascara
