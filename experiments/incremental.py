"""Escada incremental de modelos: do TF-IDF mais simples ao ensemble com BERT.

Cada degrau muda uma única coisa em relação ao melhor modelo acumulado.
A mudança só é mantida se melhorar a acurácia média e vencer na maioria
dos folds (os folds são os mesmos em todos os degraus).

    python -m experiments.incremental            # todos os degraus
    python -m experiments.incremental --ate 3    # até o degrau 3

Resultados em results/incremental/ (resultados por config ficam em cache).
"""

import argparse
import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import CLASSES, FINETUNE_HISTORY_FILE, FINETUNE_HOLDOUT_PROBS_FILE, RANDOM_STATE
from src.data import carregar_dados_treino
from src.ensemble import aplicar_lookup, combinar, melhores_pesos, tabela_lookup
from src.incremental import (
    CONFIG_INICIAL,
    INCREMENTAL_DIR,
    avaliar_config,
    avaliar_holdout,
    diagnosticar,
    indices_holdout,
    matriz_confusao,
    splits_agrupados,
    splits_padrao,
)
from src.results import registrar_experimento, salvar_tabela


FOLDS_PARA_MANTER = 3


@dataclass
class Degrau:
    numero: str
    descricao: str
    hipotese: str
    candidatos: Callable[[dict], list[dict]] = field(default=lambda c: [])


def _com(config: dict, **mudancas) -> dict:
    nova = copy.deepcopy(config)
    for chave, valor in mudancas.items():
        if "__" in chave:
            bloco, sub = chave.split("__")
            nova[bloco][sub] = valor
        else:
            nova[chave] = valor
    return nova


CHAR_PADRAO = {"analyzer": "char", "ngram_range": (4, 6), "min_df": 2, "sublinear_tf": True}

DEGRAUS = [
    Degrau("1", "sublinear_tf=True", "Textos longos dominam pela contagem bruta de termos?",
           lambda c: [_com(c, word__sublinear_tf=True)]),
    Degrau("2", "min_df", "Vocabulário raro é ruído?",
           lambda c: [_com(c, word__min_df=m) for m in (2, 3, 5, 10)]),
    Degrau("3", "bigramas (1,2)", "Expressões de duas palavras carregam sinal de clareza?",
           lambda c: [_com(c, word__ngram_range=(1, 2))]),
    Degrau("4", "ajuste de C", "A regularização padrão está adequada (gap treino/val)?",
           lambda c: [_com(c, C=v) for v in (0.25, 0.5, 2.0)]),
    Degrau("5", "+ TF-IDF de caracteres (4-6)", "Morfologia e grafia ajudam além das palavras?",
           lambda c: [_com(c, char=CHAR_PADRAO)]),
    Degrau("6", "+ características estruturais", "Tamanho/formato explicam a clareza?",
           lambda c: [_com(c, structural_weight=w) for w in (0.25, 0.5, 1.0)]),
    Degrau("7", "normalização de URLs e números", "Números/URLs específicos atrapalham a generalização?",
           lambda c: [_com(c, preprocessor="urls_numeros")]),
]


# ============================================================
# AUXILIARES
# ============================================================

def resumo_linha(numero, descricao, hipotese, resultado, anterior, holdout, protocolo) -> dict:
    vitorias = int((resultado["acc_folds"] > anterior["acc_folds"]).sum()) if anterior else None
    delta = resultado["accuracy"] - anterior["accuracy"] if anterior else None
    return {
        "degrau": numero,
        "mudanca": descricao,
        "hipotese": hipotese,
        "protocolo": protocolo,
        "accuracy": resultado["accuracy"],
        "accuracy_std": resultado["accuracy_std"],
        "f1_macro": resultado["f1_macro"],
        "train_accuracy": resultado.get("train_accuracy"),
        "accuracy_gap": resultado.get("accuracy_gap"),
        "delta": delta,
        "folds_vencidos": vitorias,
        "holdout_accuracy": holdout,
    }


def salvar_diagnostico(numero, X, y, pred, mascara=None) -> pd.DataFrame:
    diag = diagnosticar(X, y, pred, mascara)
    salvar_tabela(diag, INCREMENTAL_DIR / f"diagnostico_{numero}.csv")
    y_m = np.asarray(y) if mascara is None else np.asarray(y)[mascara]
    p_m = np.asarray(pred) if mascara is None else np.asarray(pred)[mascara]
    matriz_confusao(y_m, p_m).to_csv(INCREMENTAL_DIR / f"confusao_{numero}.csv")
    return diag


def imprimir_degrau(linha, diag, config=None) -> None:
    print("\n" + "=" * 60)
    print(f"DEGRAU {linha['degrau']}: {linha['mudanca']}")
    print("=" * 60)
    print(f"Hipótese: {linha['hipotese']}")
    std = f" ± {linha['accuracy_std']:.4f}" if linha.get("accuracy_std") is not None else ""
    print(f"Accuracy: {linha['accuracy']:.4f}{std} ({linha['protocolo']})")
    if linha["delta"] is not None:
        vencidos = linha.get("folds_vencidos")
        trecho_folds = f" | folds vencidos: {vencidos}/5" if vencidos is not None else ""
        print(f"Delta: {linha['delta']:+.4f}{trecho_folds} | mantido: {linha.get('mantido')}")
    if linha.get("holdout_accuracy") is not None:
        print(f"Holdout: {linha['holdout_accuracy']:.4f}")
    if diag is not None:
        tabela = diag.pivot_table(index="grupo", columns="valor", values="accuracy", sort=False)
        print(tabela.round(3).to_string())
    if config is not None:
        print(f"Config: {config}")


def registrar(linha, config) -> None:
    registrar_experimento(
        {
            "experiment_id": f"incremental_{linha['degrau']}",
            "experiment_name": f"Incremental {linha['degrau']}: {linha['mudanca']}",
            "category": "incremental",
            "representation": "TF-IDF incremental",
            "model": "Regressão Logística",
            "evaluation": linha["protocolo"],
            "folds": 5,
            "accuracy": linha["accuracy"],
            "accuracy_std": linha["accuracy_std"],
            "f1_macro": linha["f1_macro"],
            "train_accuracy": linha["train_accuracy"],
            "accuracy_gap": linha["accuracy_gap"],
            "parameters": config,
            "notes": f"delta={linha['delta']}; folds_vencidos={linha['folds_vencidos']}; mantido={linha.get('mantido')}; holdout={linha['holdout_accuracy']}",
        }
    )


def acc_lookup_cv(config, X, y, splits, nome_cache) -> dict:
    """CV com lookup de textos repetidos (tabela construída só com o treino do fold)."""
    base = avaliar_config(config, X, y, splits, nome_cache=nome_cache)
    X_arr, y_arr = np.asarray(X, dtype=object), np.asarray(y)
    pred = base["pred"].copy()
    for treino, validacao in splits:
        tabela = tabela_lookup(X_arr[treino], y_arr[treino])
        pred[validacao], _ = aplicar_lookup(X_arr[validacao], pred[validacao], tabela)
    acc_folds = np.array([accuracy_score(y_arr[base["fold_ids"] == f], pred[base["fold_ids"] == f]) for f in range(len(splits))])
    return {**base, "pred": pred, "acc_folds": acc_folds, "accuracy": float(acc_folds.mean()),
            "accuracy_std": float(acc_folds.std()), "f1_macro": float(f1_score(y_arr, pred, average="macro"))}


# ============================================================
# EXECUÇÃO
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ate", type=int, default=99)
    args = parser.parse_args()

    X, y = carregar_dados_treino()
    X_arr, y_arr = np.asarray(X, dtype=object), np.asarray(y)
    agrupados = splits_agrupados(X_arr, y_arr)
    padrao = splits_padrao(X_arr, y_arr)
    linhas, candidatos = [], []

    # ---------- Degrau 0a: classe majoritária ----------
    _, validacao = indices_holdout(y_arr)
    majoritaria = pd.Series(y_arr).value_counts().idxmax()
    acc_maj = float((y_arr == majoritaria).mean())
    linhas.append({"degrau": "0a", "mudanca": f"classe majoritária ({majoritaria})", "hipotese": "Piso",
                   "protocolo": "proporção", "accuracy": acc_maj, "accuracy_std": 0.0,
                   "holdout_accuracy": float((y_arr[validacao] == majoritaria).mean())})
    print(f"\nDEGRAU 0a: classe majoritária = {acc_maj:.4f}")

    # ---------- Degrau 0b: baseline oficial, CV padrão vs agrupado ----------
    config = copy.deepcopy(CONFIG_INICIAL)
    melhor = avaliar_config(config, X_arr, y_arr, agrupados, nome_cache="agrupado")
    padrao_base = avaliar_config(config, X_arr, y_arr, padrao, nome_cache="padrao")
    holdout = avaliar_holdout(config, X_arr, y_arr)["accuracy"]
    linha = resumo_linha("0b", "TF-IDF default + LR (baseline oficial)", "Ponto de partida; vazamento por duplicatas",
                         melhor, None, holdout, "5-fold agrupado por texto")
    linha["mantido"] = True
    linha["accuracy_cv_padrao"] = padrao_base["accuracy"]
    linhas.append(linha)
    diag = salvar_diagnostico("0b", X_arr, y_arr, melhor["pred"])
    imprimir_degrau(linha, diag, config)
    print(f"CV padrão (com vazamento): {padrao_base['accuracy']:.4f} | "
          f"vazamento: {padrao_base['accuracy'] - melhor['accuracy']:+.4f}")
    registrar(linha, config)

    # ---------- Degraus 1..7: uma mudança por vez ----------
    for degrau in DEGRAUS:
        if int(degrau.numero) > args.ate:
            break
        resultados = []
        for candidato in degrau.candidatos(config):
            r = avaliar_config(candidato, X_arr, y_arr, agrupados, nome_cache="agrupado")
            resultados.append((candidato, r))
            candidatos.append({"degrau": degrau.numero, "config": str(candidato), "accuracy": r["accuracy"],
                               "accuracy_std": r["accuracy_std"], "delta": r["accuracy"] - melhor["accuracy"],
                               "folds_vencidos": int((r["acc_folds"] > melhor["acc_folds"]).sum())})
        candidato, r = max(resultados, key=lambda item: item[1]["accuracy"])
        holdout = avaliar_holdout(candidato, X_arr, y_arr)["accuracy"]
        linha = resumo_linha(degrau.numero, degrau.descricao, degrau.hipotese, r, melhor, holdout, "5-fold agrupado por texto")
        linha["mantido"] = bool(linha["delta"] > 0 and linha["folds_vencidos"] >= FOLDS_PARA_MANTER)
        linhas.append(linha)
        diag = salvar_diagnostico(degrau.numero, X_arr, y_arr, r["pred"])
        imprimir_degrau(linha, diag, candidato)
        registrar(linha, candidato)
        if linha["mantido"]:
            config, melhor = candidato, r

    # ---------- Degrau 8: lookup de textos repetidos (exige CV padrão) ----------
    if args.ate >= 8:
        sem = avaliar_config(config, X_arr, y_arr, padrao, nome_cache="padrao")
        com = acc_lookup_cv(config, X_arr, y_arr, padrao, nome_cache="padrao")
        treino, validacao = indices_holdout(y_arr)
        ho = avaliar_holdout(config, X_arr, y_arr)
        pred_ho, _ = aplicar_lookup(X_arr[validacao], ho["pred"][validacao], tabela_lookup(X_arr[treino], y_arr[treino]))
        linha = resumo_linha("8", "lookup de textos repetidos", "Rótulo de textos já vistos no treino é mais confiável?",
                             com, sem, float(accuracy_score(y_arr[validacao], pred_ho)), "5-fold padrão (lookup exige textos repetidos entre folds)")
        linha["mantido"] = bool(linha["delta"] > 0 and linha["folds_vencidos"] >= FOLDS_PARA_MANTER)
        linha["accuracy_cv_padrao"] = sem["accuracy"]
        linhas.append(linha)
        diag = salvar_diagnostico("8", X_arr, y_arr, com["pred"])
        imprimir_degrau(linha, diag, {**config, "lookup": True})
        registrar(linha, {**config, "lookup": True})
        config = {**config, "lookup": linha["mantido"]}

    # ---------- Degrau 9: ensemble com BERT fine-tuned (holdout) ----------
    if args.ate >= 9 and FINETUNE_HOLDOUT_PROBS_FILE.exists():
        treino, validacao = indices_holdout(y_arr)
        ho = avaliar_holdout(config, X_arr, y_arr)
        probs_classico = ho["probs"][validacao]
        historico = pd.read_csv(FINETUNE_HISTORY_FILE)
        epoca = int(historico.loc[historico["val_accuracy"].idxmax(), "epoch"])
        df_bert = pd.read_csv(FINETUNE_HOLDOUT_PROBS_FILE)
        df_bert = df_bert[df_bert["epoch"] == epoca].set_index("index").loc[validacao]
        probs = {"classico": probs_classico, "bert": df_bert[[f"p_{c}" for c in CLASSES]].to_numpy()}
        y_val = y_arr[validacao]

        def acc_com(pred):
            if config["lookup"]:
                pred, _ = aplicar_lookup(X_arr[validacao], pred, tabela_lookup(X_arr[treino], y_arr[treino]))
            return float(accuracy_score(y_val, pred)), pred

        acc_classico, _ = acc_com(np.array(CLASSES)[probs_classico.argmax(1)])
        acc_bert, _ = acc_com(np.array(CLASSES)[probs["bert"].argmax(1)])

        # Pesos escolhidos em uma metade do holdout e avaliados na outra.
        from sklearn.model_selection import StratifiedKFold
        metades = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y_val)), y_val)
        pred_honesto = np.empty(len(y_val), dtype=object)
        pesos_metades = []
        for sel, aval in metades:
            pesos, _ = melhores_pesos({k: v[sel] for k, v in probs.items()}, y_val[sel])
            pesos_metades.append(pesos)
            pred_honesto[aval] = np.array(CLASSES)[combinar({k: v[aval] for k, v in probs.items()}, pesos).argmax(1)]
        acc_ens, pred_final = acc_com(pred_honesto.astype(str))
        pesos_finais, _ = melhores_pesos(probs, y_val)

        linha = {"degrau": "9", "mudanca": "+ BERT fine-tuned (ensemble de probabilidades)",
                 "hipotese": "Semântica contextual corrige erros do modelo lexical?",
                 "protocolo": "holdout 20% (pesos escolhidos em metades cruzadas)",
                 "accuracy": acc_ens, "accuracy_std": None,
                 "f1_macro": float(f1_score(y_val, pred_final, average="macro")),
                 "delta": acc_ens - acc_classico, "holdout_accuracy": acc_ens, "mantido": acc_ens > acc_classico}
        linhas.append(linha)
        mascara = np.zeros(len(y_arr), dtype=bool)
        mascara[validacao] = True
        pred_completo = np.empty(len(y_arr), dtype=object)
        pred_completo[validacao] = pred_final
        diag = salvar_diagnostico("9", X_arr, y_arr, pred_completo, mascara)
        imprimir_degrau(linha, diag)
        print(f"Holdout: clássico={acc_classico:.4f} | BERT sozinho={acc_bert:.4f} | ensemble={acc_ens:.4f}")
        print(f"Pesos por metade: {pesos_metades} | pesos finais: {pesos_finais}")
        registrar({**linha, "train_accuracy": None, "accuracy_gap": None, "folds_vencidos": None}, {"pesos": pesos_finais})

    salvar_tabela(pd.DataFrame(linhas), INCREMENTAL_DIR / "degraus.csv")
    salvar_tabela(pd.DataFrame(candidatos), INCREMENTAL_DIR / "candidatos.csv")
    print("\n" + pd.DataFrame(linhas)[["degrau", "mudanca", "accuracy", "delta", "folds_vencidos", "mantido", "holdout_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
