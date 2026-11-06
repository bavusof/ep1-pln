"""Probabilidades out-of-fold (OOF) e de teste para o stacking do EP2.

Todos os modelos de base usam os MESMOS folds (5-fold StratifiedGroupKFold
agrupado por texto, random_state=42). Para cada base salvamos:
- `oof`: probabilidades do treino, cada linha prevista pelo modelo do fold
  em que ela ficou de fora;
- `test1` / `test2`: média das probabilidades dos 5 modelos dos folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARQUIVO_TESTE1_ROTULADO,
    ARQUIVO_TESTE2,
    CLASSES,
    COLUNA_TEXTO,
    OOF_DIR,
)
from src.incremental import splits_agrupados


def carregar_textos_teste() -> dict[str, pd.Series]:
    """Somente os textos dos testes; o gabarito do teste 1 não é lido aqui."""
    return {
        nome: pd.read_excel(caminho)[COLUNA_TEXTO].fillna("").astype(str)
        for nome, caminho in (("test1", ARQUIVO_TESTE1_ROTULADO), ("test2", ARQUIVO_TESTE2))
    }


def folds_ep2(X, y) -> list[tuple[np.ndarray, np.ndarray]]:
    return splits_agrupados(np.asarray(X, dtype=object), np.asarray(y))


def arquivo_fold(tag: str, fold: int) -> Path:
    return OOF_DIR / f"{tag}_fold{fold}.npz"


def arquivo_base(tag: str) -> Path:
    return OOF_DIR / f"{tag}.npz"


def salvar_fold(tag: str, fold: int, *, val_idx, probs_val, probs_test1, probs_test2) -> None:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        arquivo_fold(tag, fold),
        val_idx=val_idx,
        probs_val=probs_val,
        probs_test1=probs_test1,
        probs_test2=probs_test2,
    )


def consolidar(tag: str, n_treino: int, n_folds: int) -> dict[str, np.ndarray]:
    """Junta os arquivos por fold em um único arquivo da base."""
    oof = np.full((n_treino, len(CLASSES)), np.nan)
    fold_ids = np.full(n_treino, -1)
    t1, t2 = [], []
    for fold in range(n_folds):
        dados = np.load(arquivo_fold(tag, fold))
        oof[dados["val_idx"]] = dados["probs_val"]
        fold_ids[dados["val_idx"]] = fold
        t1.append(dados["probs_test1"])
        t2.append(dados["probs_test2"])
    if np.isnan(oof).any():
        raise ValueError(f"OOF incompleto para {tag}: há linhas sem previsão.")
    base = {"oof": oof, "fold_ids": fold_ids, "test1": np.mean(t1, axis=0), "test2": np.mean(t2, axis=0)}
    np.savez(arquivo_base(tag), **base)
    return base


def carregar_base(tag: str) -> dict[str, np.ndarray]:
    dados = np.load(arquivo_base(tag))
    return {chave: dados[chave] for chave in dados.files}


def acuracia_por_fold(oof, fold_ids, y) -> np.ndarray:
    y = np.asarray(y)
    pred = np.array(CLASSES)[oof.argmax(1)]
    return np.array([(pred[fold_ids == f] == y[fold_ids == f]).mean() for f in np.unique(fold_ids)])
