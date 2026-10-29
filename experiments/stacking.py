"""Stacking do EP2: escolhe a combinação das bases só com o treino.

Estratégias comparadas nos MESMOS 5 folds agrupados das bases (CV externa):
- cada base sozinha;
- média simples das bases;
- média ponderada (pesos escolhidos no treino de cada fold externo);
- meta-Regressão Logística sobre log-probabilidades + atributos de texto
  (C escolhido por CV interna de 3 folds agrupados dentro do fold externo).

Depois, a regra de lookup de duplicatas é avaliada por leave-one-out.
A estratégia vencedora é reajustada em todo o OOF e salva para o
`experiments/predict_ep2.py`.

    python -m experiments.stacking [base1 base2 ...]
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import CLASSES, EP2_RESULTS_DIR, OOF_DIR, RANDOM_STATE, STACKING_CONFIG_FILE
from src.data import carregar_dados_treino
from src.incremental import grupos_por_texto
from src.oof import carregar_base
from src.results import registrar_experimento, salvar_tabela
from src.stacking import (
    ajustar_media_ponderada,
    aplicar_media_ponderada,
    aplicar_regra_lookup,
    criar_meta_lr,
    lookup_treino,
    montar_atributos,
)

GRADE_C = [0.01, 0.03, 0.1, 0.3, 1.0]
REGRAS_LOOKUP = [(1, 1.0), (2, 1.0), (2, 0.67), (3, 0.6), (5, 0.5)]
META_MODELO_FILE = EP2_RESULTS_DIR / "meta_lr.joblib"


def bases_disponiveis() -> list[str]:
    return sorted(p.stem for p in OOF_DIR.glob("*.npz") if "_fold" not in p.stem and "_amostra" not in p.stem)


def escolher_C(Z, y, grupos) -> float:
    """C da meta-LR por CV interna agrupada (só com o treino do fold externo)."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    medias = []
    for C in GRADE_C:
        accs = [
            (criar_meta_lr(C).fit(Z[tr], y[tr]).predict(Z[va]) == y[va]).mean()
            for tr, va in cv.split(Z, y, grupos)
        ]
        medias.append(np.mean(accs))
    return GRADE_C[int(np.argmax(medias))]


def main() -> None:
    nomes = sys.argv[1:] or bases_disponiveis()
    X, y = carregar_dados_treino()
    X_arr, y_arr = X.to_numpy(dtype=object), y.to_numpy()
    grupos = grupos_por_texto(X_arr)
    bases = {nome: carregar_base(nome) for nome in nomes}

    fold_ids = bases[nomes[0]]["fold_ids"]
    for nome, base in bases.items():
        assert (base["fold_ids"] == fold_ids).all(), f"{nome} usa folds diferentes"
    oof = {nome: base["oof"] for nome, base in bases.items()}
    Z = montar_atributos(oof, X_arr)
    folds = sorted(np.unique(fold_ids))
    print(f"Bases: {nomes}")

    # ---------- CV externa ----------
    preds = {f"base:{n}": np.array(CLASSES)[p.argmax(1)] for n, p in oof.items()}
    preds["media_simples"] = np.array(CLASSES)[(sum(oof.values()) / len(oof)).argmax(1)]
    preds["media_ponderada"] = np.empty(len(y_arr), dtype=object)
    preds["meta_lr"] = np.empty(len(y_arr), dtype=object)
    escolhas = {"media_ponderada": [], "meta_lr": []}
    for f in folds:
        tr, va = fold_ids != f, fold_ids == f
        pesos = ajustar_media_ponderada({n: p[tr] for n, p in oof.items()}, y_arr[tr])
        preds["media_ponderada"][va] = aplicar_media_ponderada({n: p[va] for n, p in oof.items()}, pesos)
        C = escolher_C(Z[tr], y_arr[tr], grupos[tr])
        preds["meta_lr"][va] = criar_meta_lr(C).fit(Z[tr], y_arr[tr]).predict(Z[va])
        escolhas["media_ponderada"].append(pesos)
        escolhas["meta_lr"].append(C)

    linhas = []
    for estrategia, pred in preds.items():
        accs = np.array([(pred[fold_ids == f] == y_arr[fold_ids == f]).mean() for f in folds])
        linhas.append({"estrategia": estrategia, "accuracy": accs.mean(), "accuracy_std": accs.std(),
                       "por_fold": np.round(accs, 4).tolist()})
    tabela = pd.DataFrame(linhas).sort_values("accuracy", ascending=False)
    print("\n" + tabela.to_string(index=False))
    print(f"\nPesos por fold: {escolhas['media_ponderada']}\nC por fold: {escolhas['meta_lr']}")

    vencedora = tabela.iloc[0]["estrategia"]
    pred_vencedora = preds[vencedora]
    base_unica = vencedora.split(":", 1)[1] if vencedora.startswith("base:") else None

    # ---------- Regra de lookup (leave-one-out sobre cópias idênticas) ----------
    lookup = lookup_treino(X_arr, y_arr)
    tem_copia = np.expm1(lookup[:, 3]) >= 1
    base_acc = (pred_vencedora == y_arr).mean()
    regras = []
    for min_copias, pureza in REGRAS_LOOKUP:
        pred_lk, usado = aplicar_regra_lookup(pred_vencedora, lookup, min_copias=min_copias, pureza=pureza)
        regras.append({"min_copias": min_copias, "pureza": pureza, "linhas_alteradas": int(usado.sum()),
                       "acc_nas_linhas_modelo": float((pred_vencedora[usado] == y_arr[usado]).mean()) if usado.any() else None,
                       "acc_nas_linhas_lookup": float((pred_lk[usado] == y_arr[usado]).mean()) if usado.any() else None,
                       "accuracy_total": float((pred_lk == y_arr).mean()), "delta": float((pred_lk == y_arr).mean() - base_acc)})
    tabela_lookup = pd.DataFrame(regras)
    print(f"\nLookup (LOO; {tem_copia.sum()} linhas têm cópia no treino) sobre '{vencedora}':")
    print(tabela_lookup.to_string(index=False))
    melhor_regra = tabela_lookup.sort_values("delta", ascending=False).iloc[0]
    usar_lookup = bool(melhor_regra["delta"] > 0)

    # ---------- Ajuste final em todo o OOF ----------
    config = {"bases": nomes, "estrategia": vencedora,
              "lookup": {"usar": usar_lookup, "min_copias": int(melhor_regra["min_copias"]), "pureza": float(melhor_regra["pureza"])},
              "cv_accuracy": float(tabela.set_index("estrategia").loc[vencedora, "accuracy"])}
    if base_unica is not None:
        # Uma base sozinha venceu: equivale a média ponderada com peso 1 nela.
        config["estrategia"] = "media_ponderada"
        config["pesos"] = {nome: float(nome == base_unica) for nome in nomes}
    elif vencedora == "media_simples":
        config["estrategia"] = "media_ponderada"
        config["pesos"] = {nome: 1.0 / len(nomes) for nome in nomes}
    elif vencedora == "media_ponderada":
        config["pesos"] = ajustar_media_ponderada(oof, y_arr)
    elif vencedora == "meta_lr":
        C = escolher_C(Z, y_arr, grupos)
        config["C"] = C
        EP2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(criar_meta_lr(C).fit(Z, y_arr), META_MODELO_FILE)

    EP2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STACKING_CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    salvar_tabela(tabela, EP2_RESULTS_DIR / "stacking_cv.csv")
    salvar_tabela(tabela_lookup, EP2_RESULTS_DIR / "lookup_loo.csv")
    print(f"\nEscolhida: {vencedora} | lookup: {config['lookup']} | config salva em {STACKING_CONFIG_FILE}")

    registrar_experimento(
        {
            "experiment_id": "ep2_stacking",
            "experiment_name": f"EP2 stacking ({vencedora})",
            "category": "ep2_stacking",
            "representation": " + ".join(nomes),
            "model": vencedora,
            "evaluation": "CV externa nos 5 folds agrupados das bases (OOF)",
            "folds": len(folds),
            "accuracy": config["cv_accuracy"],
            "accuracy_std": float(tabela.set_index("estrategia").loc[vencedora, "accuracy_std"]),
            "parameters": config,
            "notes": "Estratégia e hiperparâmetros escolhidos só no treino; testes não usados.",
        }
    )


if __name__ == "__main__":
    main()
