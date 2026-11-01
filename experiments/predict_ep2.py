"""Gera as planilhas do EP2 a partir do stacking escolhido em experiments.stacking.

- results/ep2/test2_rotulado.xlsx: ENTREGA oficial (897 linhas).
- results/ep2/test1_ep2.xlsx: rótulos do EP2 no teste 1, usados só no McNemar.

O gabarito do teste 1 não é lido aqui (somente a coluna de texto).

    python -m experiments.predict_ep2
"""

import json

import joblib
import numpy as np
import pandas as pd

from experiments.stacking import META_MODELO_FILE
from src.config import (
    ARQUIVO_TESTE1_EP2,
    ARQUIVO_TESTE1_ROTULADO,
    ARQUIVO_TESTE2,
    ARQUIVO_TESTE2_ROTULADO,
    CLASSES,
    COLUNA_CLASSE,
    COLUNA_TEXTO,
    STACKING_CONFIG_FILE,
)
from src.atributos_extras import carregar_extras
from src.data import carregar_dados_treino
from src.oof import carregar_base
from src.stacking import aplicar_media_ponderada, aplicar_regra_lookup, lookup_teste, montar_atributos

TAMANHOS_ESPERADOS = {"test1": 900, "test2": 897}


def prever(config, nome_teste, textos, X_treino, y_treino) -> tuple[np.ndarray, int]:
    probs = {nome: carregar_base(nome)[nome_teste] for nome in config["bases"]}
    if config["estrategia"] == "meta_lr":
        extras = carregar_extras(config.get("extras", []), textos)
        pred = joblib.load(META_MODELO_FILE).predict(montar_atributos(probs, textos, extras))
    else:
        pred = aplicar_media_ponderada(probs, config["pesos"])
    alterados = 0
    if config["lookup"]["usar"]:
        pred, usado = aplicar_regra_lookup(
            pred,
            lookup_teste(X_treino, y_treino, textos),
            min_copias=config["lookup"]["min_copias"],
            pureza=config["lookup"]["pureza"],
        )
        alterados = int(usado.sum())
    return np.asarray(pred, dtype=object), alterados


def salvar_e_validar(textos: pd.Series, pred, destino, esperado: int) -> None:
    saida = pd.DataFrame({COLUNA_TEXTO: textos.to_numpy(), COLUNA_CLASSE: pred})
    destino.parent.mkdir(parents=True, exist_ok=True)
    saida.to_excel(destino, index=False)

    relido = pd.read_excel(destino)
    assert list(relido.columns) == [COLUNA_TEXTO, COLUNA_CLASSE], f"colunas: {list(relido.columns)}"
    assert len(relido) == esperado, f"{len(relido)} linhas, esperado {esperado}"
    assert (relido[COLUNA_TEXTO].fillna("").astype(str).to_numpy() == textos.fillna("").astype(str).to_numpy()).all(), "ordem dos textos alterada"
    assert relido[COLUNA_CLASSE].isin(CLASSES).all(), "rótulo fora de {c1, c234, c5}"
    print(f"{destino.name}: OK ({len(relido)} linhas) {relido[COLUNA_CLASSE].value_counts().to_dict()}")


def main() -> None:
    config = json.loads(STACKING_CONFIG_FILE.read_text(encoding="utf-8"))
    print(f"Stacking: {config['estrategia']} | bases: {config['bases']} | lookup: {config['lookup']}")
    X, y = carregar_dados_treino()

    for nome, entrada, destino in (
        ("test2", ARQUIVO_TESTE2, ARQUIVO_TESTE2_ROTULADO),
        ("test1", ARQUIVO_TESTE1_ROTULADO, ARQUIVO_TESTE1_EP2),
    ):
        textos = pd.read_excel(entrada, usecols=[COLUNA_TEXTO])[COLUNA_TEXTO]
        pred, alterados = prever(config, nome, textos.fillna("").astype(str), X, y)
        print(f"{nome}: {alterados} linhas rotuladas pelo lookup")
        salvar_e_validar(textos, pred, destino, TAMANHOS_ESPERADOS[nome])


if __name__ == "__main__":
    main()
