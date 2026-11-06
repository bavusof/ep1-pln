"""Confident learning (cleanlab) contra o ruído de rótulo, sem vazamento.

Para cada fold externo, a qualidade do rótulo de cada linha de TREINO é
estimada com uma CV interna (5 folds agrupados por texto) usando só essas
linhas. Assim os rótulos da validação do fold nunca influenciam quais linhas
de treino são descartadas. As linhas com menor qualidade são removidas.
"""

import copy
from pathlib import Path

import numpy as np
from cleanlab.rank import get_label_quality_scores
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.config import CLASSES, N_SPLITS, OOF_DIR, RANDOM_STATE
from src.incremental import CONFIG_INICIAL, construir_pipeline, grupos_por_texto

# Mesma configuração do degrau 6 da escada (bigramas + características estruturais).
CONFIG_PONTUACAO = copy.deepcopy(CONFIG_INICIAL)
CONFIG_PONTUACAO["word"]["ngram_range"] = (1, 2)
CONFIG_PONTUACAO["structural_weight"] = 1.0


def arquivo_pontuacao(fold: int) -> Path:
    return OOF_DIR / f"ruido_qualidade_fold{fold}.npy"


def pontuar_qualidade(textos, rotulos, *, fold: int | None = None) -> np.ndarray:
    """Qualidade do rótulo (0 = provável ruído, 1 = confiável) de cada linha.

    `fold` ativa o cache em disco (depende só das linhas de treino do fold)."""
    cache = arquivo_pontuacao(fold) if fold is not None else None
    if cache is not None and cache.exists():
        qualidade = np.load(cache)
        if len(qualidade) == len(textos):
            return qualidade

    textos = np.asarray(textos, dtype=object)
    rotulos = np.asarray(rotulos)
    indices = np.array([CLASSES.index(r) for r in rotulos])
    probs = np.zeros((len(rotulos), len(CLASSES)))
    modelo = construir_pipeline(CONFIG_PONTUACAO)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in cv.split(textos, rotulos, groups=grupos_por_texto(textos)):
        ajustado = clone(modelo).fit(textos[tr], rotulos[tr])
        ordem = [list(ajustado.classes_).index(c) for c in CLASSES]
        probs[va] = ajustado.predict_proba(textos[va])[:, ordem]

    qualidade = get_label_quality_scores(indices, probs)
    if cache is not None:
        OOF_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, qualidade)
    return qualidade


def mascara_manter(textos, rotulos, frac: float, *, fold: int | None = None) -> tuple[np.ndarray, dict]:
    """Máscara das linhas a manter após remover a fração `frac` de menor qualidade."""
    qualidade = pontuar_qualidade(textos, rotulos, fold=fold)
    corte = np.quantile(qualidade, frac)
    manter = qualidade > corte
    rotulos = np.asarray(rotulos)
    relatorio = {
        "linhas_iniciais": int(len(rotulos)),
        "removidas": int((~manter).sum()),
        "removidas_por_classe": {c: int(((~manter) & (rotulos == c)).sum()) for c in CLASSES},
    }
    return manter, relatorio
