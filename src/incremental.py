"""Avaliação incremental: um modelo base e uma mudança por degrau.

Todos os degraus usam o mesmo protocolo:
- 5-fold StratifiedGroupKFold agrupado pelo texto (textos idênticos nunca
  ficam ao mesmo tempo no treino e na validação);
- predições out-of-fold para o diagnóstico de erros;
- holdout estratificado de 20% (mesmo split do baseline oficial) para
  comparar com o BERT fine-tuned.
"""

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import CLASSES, N_SPLITS, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from src.features import TextStructuralFeatures
from src.models import (
    OrdinalClarityClassifier,
    criar_regressao_logistica,
    criar_tfidf_caracteres,
    criar_tfidf_word,
)
from src.text_preprocessing import normalizar_urls_numeros


INCREMENTAL_DIR = RESULTS_DIR / "incremental"
CACHE_DIR = INCREMENTAL_DIR / "cache"

PREPROCESSADORES = {
    None: None,
    "urls_numeros": normalizar_urls_numeros,
}

FAIXAS_TAMANHO = [0, 50, 150, 300, np.inf]
ROTULOS_TAMANHO = ["<=50", "51-150", "151-300", ">300"]


# ============================================================
# CONSTRUÇÃO DO PIPELINE A PARTIR DA CONFIGURAÇÃO ACUMULADA
# ============================================================

CONFIG_INICIAL = {
    "word": {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0, "sublinear_tf": False},
    "preprocessor": None,
    "char": None,
    "structural_weight": None,
    "C": 1.0,
    "class_weight": None,
    "selection_percentile": None,
    "lookup": False,
}


def construir_features(config: Mapping[str, Any]) -> FeatureUnion:
    """Monta apenas a representação de atributos (sem o classificador).

    Reutilizado pelo pipeline padrão, pelos membros do ensemble clássico e
    pela decomposição ordinal, garantindo a mesma representação em todos.
    """
    preprocessor = PREPROCESSADORES[config["preprocessor"]]
    blocos = [
        ("word", criar_tfidf_word(**config["word"], preprocessor=preprocessor)),
    ]
    pesos = {}
    if config["char"] is not None:
        blocos.append(("char", criar_tfidf_caracteres(**config["char"], preprocessor=preprocessor)))
    if config["structural_weight"] is not None:
        blocos.append(
            (
                "structural",
                Pipeline([("extract", TextStructuralFeatures()), ("scale", StandardScaler(with_mean=False))]),
            )
        )
        pesos["structural"] = config["structural_weight"]
    return FeatureUnion(blocos, transformer_weights=pesos or None)


def construir_pipeline(config: Mapping[str, Any]) -> Pipeline:
    etapas = [("features", construir_features(config))]
    if config.get("selection_percentile") is not None:
        etapas.append(("selection", SelectPercentile(chi2, percentile=config["selection_percentile"])))
    etapas.append(("logistic", criar_regressao_logistica(C=config["C"], class_weight=config.get("class_weight"))))
    return Pipeline(etapas)


def construir_ordinal(config: Mapping[str, Any]) -> OrdinalClarityClassifier:
    """Decomposição ordinal sobre a mesma representação de atributos."""
    return OrdinalClarityClassifier(
        construir_features(config),
        C=config["C"],
        class_weight=config.get("class_weight"),
    )


def membros_ensemble_classico(config: Mapping[str, Any], *, C_svc: float = 0.25) -> dict[str, Pipeline]:
    """LR, LinearSVC calibrado e ComplementNB sobre a mesma representação."""
    def pipe(head):
        return Pipeline([("features", construir_features(config)), ("clf", head)])

    return {
        "lr": pipe(criar_regressao_logistica(C=config["C"], class_weight=config.get("class_weight"))),
        "svc": pipe(CalibratedClassifierCV(LinearSVC(C=C_svc, random_state=RANDOM_STATE), cv=3)),
        "cnb": pipe(ComplementNB()),
    }


def chave_config(config: Mapping[str, Any]) -> str:
    texto = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:12]


# ============================================================
# PROTOCOLO DE AVALIAÇÃO
# ============================================================

def grupos_por_texto(X) -> np.ndarray:
    return pd.factorize(pd.Series(X).to_numpy())[0]


def splits_agrupados(X, y) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(X, y, groups=grupos_por_texto(X)))


def splits_padrao(X, y) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(X, y))


def indices_holdout(y) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    treino, validacao = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return treino, validacao


def _ajustar_fold(modelo, X, y, treino, validacao):
    modelo = clone(modelo).fit(X[treino], y[treino])
    ordem = [list(modelo.classes_).index(c) for c in CLASSES]
    probs = modelo.predict_proba(X[validacao])[:, ordem]
    acc_treino = accuracy_score(y[treino], modelo.predict(X[treino]))
    return validacao, probs, acc_treino


def avaliar_config(config, X, y, splits, *, nome_cache: str) -> dict[str, Any]:
    """Avalia uma configuração nos splits dados. Resultado em cache por config."""
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)
    arquivo = CACHE_DIR / f"{nome_cache}_{chave_config(config)}.npz"
    if arquivo.exists():
        dados = np.load(arquivo)
        probs, fold_ids, acc_treino = dados["probs"], dados["fold_ids"], dados["acc_treino"]
    else:
        modelo = construir_pipeline(config)
        saidas = Parallel(n_jobs=len(splits))(
            delayed(_ajustar_fold)(modelo, X, y, tr, va) for tr, va in splits
        )
        probs = np.zeros((len(y), len(CLASSES)))
        fold_ids = np.full(len(y), -1)
        acc_treino = []
        for fold, (validacao, p, acc) in enumerate(saidas):
            probs[validacao] = p
            fold_ids[validacao] = fold
            acc_treino.append(acc)
        acc_treino = np.array(acc_treino)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(arquivo, probs=probs, fold_ids=fold_ids, acc_treino=acc_treino)

    avaliado = fold_ids >= 0
    pred = np.array(CLASSES)[probs.argmax(1)]
    acc_folds = np.array(
        [accuracy_score(y[fold_ids == f], pred[fold_ids == f]) for f in range(len(splits))]
    )
    return {
        "probs": probs,
        "pred": pred,
        "fold_ids": fold_ids,
        "acc_folds": acc_folds,
        "accuracy": float(acc_folds.mean()),
        "accuracy_std": float(acc_folds.std()),
        "f1_macro": float(f1_score(y[avaliado], pred[avaliado], average="macro")),
        "train_accuracy": float(acc_treino.mean()),
        "accuracy_gap": float(acc_treino.mean() - acc_folds.mean()),
    }


def avaliar_holdout(config, X, y) -> dict[str, Any]:
    treino, validacao = indices_holdout(y)
    resultado = avaliar_config(config, X, y, [(treino, validacao)], nome_cache="holdout")
    resultado["indices_validacao"] = validacao
    return resultado


# ============================================================
# OOF PARA ESTIMADORES ARBITRÁRIOS (ensemble clássico e ordinal)
# ============================================================

def resumir_oof(probs, fold_ids, y, n_splits, acc_treino=None) -> dict[str, Any]:
    """Resume probabilidades out-of-fold no mesmo formato de avaliar_config."""
    y = np.asarray(y)
    avaliado = fold_ids >= 0
    pred = np.array(CLASSES)[probs.argmax(1)]
    acc_folds = np.array(
        [accuracy_score(y[fold_ids == f], pred[fold_ids == f]) for f in range(n_splits)]
    )
    resumo = {
        "probs": probs,
        "pred": pred,
        "fold_ids": fold_ids,
        "acc_folds": acc_folds,
        "accuracy": float(acc_folds.mean()),
        "accuracy_std": float(acc_folds.std()),
        "f1_macro": float(f1_score(y[avaliado], pred[avaliado], average="macro")),
    }
    if acc_treino is not None:
        resumo["train_accuracy"] = float(np.mean(acc_treino))
        resumo["accuracy_gap"] = float(np.mean(acc_treino) - acc_folds.mean())
    return resumo


def probs_oof_estimador(estimador, X, y, splits):
    """Probabilidades out-of-fold de um estimador sklearn qualquer (com predict_proba)."""
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)
    saidas = Parallel(n_jobs=len(splits))(
        delayed(_ajustar_fold)(estimador, X, y, tr, va) for tr, va in splits
    )
    probs = np.zeros((len(y), len(CLASSES)))
    fold_ids = np.full(len(y), -1)
    acc_treino = []
    for fold, (validacao, p, acc) in enumerate(saidas):
        probs[validacao] = p
        fold_ids[validacao] = fold
        acc_treino.append(acc)
    return probs, fold_ids, np.array(acc_treino)


# ============================================================
# CALIBRAÇÃO DE LIMIAR POR CLASSE (reescala as colunas de probabilidade)
# ============================================================

def _grade_simplex(k: int, passo: float = 0.05):
    n = round(1 / passo)
    for partes in product(range(n + 1), repeat=k - 1):
        if sum(partes) <= n:
            yield np.array([p / n for p in partes] + [(n - sum(partes)) / n])


def melhor_escala(probs, y, passo: float = 0.05):
    """Vetor de pesos por classe que maximiza a acurácia de argmax(w ⊙ p)."""
    y = np.asarray(y)
    melhor, melhor_acc = None, -1.0
    for w in _grade_simplex(len(CLASSES), passo):
        acc = float((np.array(CLASSES)[(probs * w).argmax(1)] == y).mean())
        if acc > melhor_acc + 1e-12:
            melhor, melhor_acc = w, acc
    return melhor, melhor_acc


def aplicar_escala(probs, w) -> np.ndarray:
    return np.array(CLASSES)[(probs * w).argmax(1)]


def escala_honesta(probs, y):
    """Escala escolhida em metade dos dados e aplicada na outra (estimativa honesta)."""
    y = np.asarray(y)
    pred = np.empty(len(y), dtype=object)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    for sel, aval in cv.split(np.zeros(len(y)), y):
        w, _ = melhor_escala(probs[sel], y[sel])
        pred[aval] = aplicar_escala(probs[aval], w)
    return float((pred == y).mean()), pred


# ============================================================
# DIAGNÓSTICO DE ERROS
# ============================================================

def diagnosticar(X, y, pred, mascara=None) -> pd.DataFrame:
    """Acurácia por classe, faixa de tamanho e duplicidade do texto."""
    X = pd.Series(np.asarray(X, dtype=object))
    y = np.asarray(y)
    pred = np.asarray(pred)
    if mascara is not None:
        X, y, pred = X[mascara].reset_index(drop=True), y[mascara], pred[mascara]

    acerto = pred == y
    linhas = []
    for classe in CLASSES:
        sel = y == classe
        linhas.append({"grupo": "recall", "valor": classe, "n": int(sel.sum()), "accuracy": float(acerto[sel].mean())})
    for classe in CLASSES:
        sel = pred == classe
        linhas.append({"grupo": "precision", "valor": classe, "n": int(sel.sum()), "accuracy": float(acerto[sel].mean()) if sel.any() else np.nan})

    tamanho = pd.cut(X.str.split().str.len().fillna(0), FAIXAS_TAMANHO, labels=ROTULOS_TAMANHO, include_lowest=True)
    for faixa in ROTULOS_TAMANHO:
        sel = (tamanho == faixa).to_numpy()
        linhas.append({"grupo": "tamanho_palavras", "valor": faixa, "n": int(sel.sum()), "accuracy": float(acerto[sel].mean())})

    duplicado = X.duplicated(keep=False).to_numpy()
    for rotulo, sel in (("inedito", ~duplicado), ("duplicado", duplicado)):
        linhas.append({"grupo": "duplicidade", "valor": rotulo, "n": int(sel.sum()), "accuracy": float(acerto[sel].mean())})

    return pd.DataFrame(linhas)


def matriz_confusao(y, pred) -> pd.DataFrame:
    matriz = confusion_matrix(y, pred, labels=CLASSES)
    return pd.DataFrame(matriz, index=[f"real_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])
