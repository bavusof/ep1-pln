"""Meta-atributos e estratégias de combinação para o stacking do EP2."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import CLASSES, RANDOM_STATE
from src.ensemble import combinar, melhores_pesos

EPS = 1e-6
PADRAO_URL = r"(?:https?://|www\.)"
PADRAO_ANEXO = r"\banex"


# ============================================================
# META-ATRIBUTOS
# ============================================================

def atributos_texto(textos) -> np.ndarray:
    """Poucas características simples do texto (tamanho, URL, menção a anexo)."""
    textos = pd.Series(textos).fillna("").astype(str)
    return np.column_stack(
        [
            np.log1p(textos.str.split().str.len().fillna(0)),
            textos.str.contains(PADRAO_URL, case=False, regex=True).astype(float),
            textos.str.contains(PADRAO_ANEXO, case=False, regex=True).astype(float),
        ]
    )


def lookup_treino(X_treino, y_treino) -> np.ndarray:
    """Distribuição de rótulos das OUTRAS cópias idênticas de cada texto do treino.

    Leave-one-out: a própria linha não entra. Reproduz a situação do teste,
    em que um texto pode já ter aparecido rotulado no treino. Colunas:
    fração c1, c234, c5 e log(1 + nº de outras cópias).
    """
    df = pd.DataFrame({"t": np.asarray(X_treino, dtype=object), "y": np.asarray(y_treino)})
    contagem = pd.crosstab(df["t"], df["y"]).reindex(columns=CLASSES, fill_value=0)
    por_linha = contagem.loc[df["t"]].to_numpy(dtype=float).copy()
    por_linha[np.arange(len(df)), [CLASSES.index(c) for c in df["y"]]] -= 1  # remove a própria linha
    return _normalizar_lookup(por_linha)


def lookup_teste(X_treino, y_treino, X_teste) -> np.ndarray:
    df = pd.DataFrame({"t": np.asarray(X_treino, dtype=object), "y": np.asarray(y_treino)})
    contagem = pd.crosstab(df["t"], df["y"]).reindex(columns=CLASSES, fill_value=0)
    por_linha = contagem.reindex(pd.Series(X_teste).fillna("").astype(str)).fillna(0).to_numpy(dtype=float)
    return _normalizar_lookup(por_linha)


def _normalizar_lookup(contagens) -> np.ndarray:
    total = contagens.sum(axis=1, keepdims=True)
    fracoes = np.divide(contagens, total, out=np.zeros_like(contagens), where=total > 0)
    return np.hstack([fracoes, np.log1p(total)])


def montar_atributos(probs_por_base: dict[str, np.ndarray], textos, extras=None) -> np.ndarray:
    """log-probabilidades de cada base + atributos simples do texto.

    O lookup de duplicatas NÃO entra aqui: no CV agrupado as cópias de um
    texto ficam sempre no mesmo fold, então o atributo seria zero no treino
    e não-zero no teste. Ele é tratado como regra pós-stacking.
    """
    blocos = [np.log(np.clip(probs_por_base[nome], EPS, 1.0)) for nome in sorted(probs_por_base)]
    blocos.append(atributos_texto(textos))
    if extras is not None:
        blocos.append(np.asarray(extras, dtype=float))  # atributos sem rótulo (src/atributos_extras.py)
    return np.hstack(blocos)


def aplicar_regra_lookup(pred, lookup, *, min_copias: int, pureza: float) -> tuple[np.ndarray, np.ndarray]:
    """Troca a predição pelo rótulo majoritário das cópias rotuladas do texto,
    quando há pelo menos `min_copias` cópias e a classe dominante tem fração >= `pureza`."""
    fracoes, n_copias = lookup[:, :3], np.expm1(lookup[:, 3])
    usar = (n_copias >= min_copias) & (fracoes.max(axis=1) >= pureza)
    pred = np.asarray(pred, dtype=object).copy()
    pred[usar] = np.array(CLASSES)[fracoes[usar].argmax(1)]
    return pred, usar


# ============================================================
# ESTRATÉGIAS (ajustadas em um conjunto, aplicadas em outro)
# ============================================================

def criar_meta_lr(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )


def ajustar_media_ponderada(probs_por_base, y):
    pesos, _ = melhores_pesos(probs_por_base, y, passo=0.05 if len(probs_por_base) <= 3 else 0.1)
    return pesos


def aplicar_media_ponderada(probs_por_base, pesos) -> np.ndarray:
    return np.array(CLASSES)[combinar(probs_por_base, pesos).argmax(1)]
