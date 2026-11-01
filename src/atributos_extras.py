"""Atributos extras do stacking que não dependem dos rótulos.

- "clareza": características feitas à mão (src/features_clareza.py);
- "llm": notas do LLM avaliador (experiments/llm_avaliador.py), lidas de um
  cache em disco indexado pelo hash do texto limpo.
"""

import hashlib

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.features_clareza import atributos_clareza
from src.text_preprocessing import limpar_para_bert

LLM_CACHE = RESULTS_DIR / "llm_features" / "qwen2.5-3b.npz"
LLM_COLUNAS = ["llm_p1", "llm_p2", "llm_p3", "llm_p4", "llm_p5", "llm_nota", "llm_entrega", "llm_dificil"]
CONJUNTOS = ("clareza", "llm")


def chave_texto(texto_limpo: str) -> str:
    return hashlib.sha1(texto_limpo.encode("utf-8", errors="replace")).hexdigest()


def carregar_cache_llm() -> dict[str, np.ndarray]:
    if not LLM_CACHE.exists():
        return {}
    dados = np.load(LLM_CACHE, allow_pickle=True)
    return dict(zip(dados["chaves"].tolist(), dados["valores"]))


def salvar_cache_llm(cache: dict[str, np.ndarray]) -> None:
    LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LLM_CACHE, chaves=np.array(list(cache), dtype=object),
             valores=np.array(list(cache.values())), colunas=np.array(LLM_COLUNAS))


def atributos_llm(textos) -> np.ndarray:
    """Matriz (n, 8) a partir do cache; erro se algum texto não foi avaliado."""
    cache = carregar_cache_llm()
    chaves = [chave_texto(limpar_para_bert(t)) for t in pd.Series(textos).fillna("").astype(str)]
    faltando = sum(k not in cache for k in chaves)
    if faltando:
        raise RuntimeError(f"{faltando} textos sem atributos do LLM; rode python -m experiments.llm_avaliador")
    return np.array([cache[k] for k in chaves], dtype=float)


def nomes_colunas(conjuntos) -> list[str]:
    nomes = []
    for conjunto in conjuntos:
        nomes += list(atributos_clareza(["x"]).columns) if conjunto == "clareza" else LLM_COLUNAS
    return nomes


def carregar_extras(conjuntos, textos) -> np.ndarray | None:
    """Concatena os conjuntos pedidos, na ordem dada. None se a lista for vazia."""
    blocos = []
    for conjunto in conjuntos:
        if conjunto not in CONJUNTOS:
            raise ValueError(f"conjunto de atributos desconhecido: {conjunto}")
        blocos.append(atributos_clareza(textos).to_numpy(dtype=float) if conjunto == "clareza" else atributos_llm(textos))
    return np.hstack(blocos) if blocos else None
