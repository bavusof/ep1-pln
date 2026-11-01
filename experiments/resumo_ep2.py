"""Tabela-resumo do EP2 para o relatório e a apresentação.

Recalcula as métricas de cada base diretamente dos arquivos OOF
(results/oof/*.npz), inclusive das tentativas interrompidas (folds parciais),
e acrescenta as linhas do stacking e dos testes. Saída:
results/ep2/resumo_ep2.csv.

    python -m experiments.resumo_ep2
"""

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import CLASSES, EP2_RESULTS_DIR, OOF_DIR, STACKING_CONFIG_FILE
from src.data import carregar_dados_treino
from src.results import salvar_tabela

DESCRICOES = {
    "tfidf": "TF-IDF palavra (1,2) + estruturais + LR (melhor clássico do EP1)",
    "tfidf_limpo": "TF-IDF com treino tratado (dedup por voto, curtos removidos)",
    "tfidf_cl10": "TF-IDF com confident learning (cleanlab, remove 10%)",
    "tfidf_cl20": "TF-IDF com confident learning (cleanlab, remove 20%)",
    "ep1_ensemble": "Modelo entregue no EP1 (TF-IDF 0,2 + híbrido Sentence-BERT 0,8)",
    "bertimbau_base": "BERTimbau-base fine-tuned (2 épocas, head+tail 256)",
    "bertimbau_base_sord": "BERTimbau-base com perda ordinal SORD",
    "bertimbau_base_limpo": "BERTimbau-base com treino tratado",
    "bertimbau_base_cl10": "BERTimbau-base com confident learning (cleanlab 10%)",
    "bertimbau_large": "BERTimbau-large fine-tuned (lr 1e-5, lote efetivo 16)",
    "albertina": "Albertina-100M PT-BR (DeBERTa) fine-tuned — divergiu em 2 folds",
    "tucano_lora": "Tucano-1b1 (LLM PT-BR) + LoRA r=16 — interrompido",
    "setfit_bertlarge_sts": "SetFit contrastivo (bert-large-pt-sts) — interrompido",
}


def metricas(pred, y, fold_ids):
    folds = sorted(f for f in np.unique(fold_ids) if f >= 0)
    acc = np.array([(pred[fold_ids == f] == y[fold_ids == f]).mean() for f in folds])
    cobertos = fold_ids >= 0
    recall = {c: float((pred[cobertos & (y == c)] == c).mean()) for c in CLASSES}
    return acc, recall, len(folds)


def main() -> None:
    _, y = carregar_dados_treino()
    y = y.to_numpy()
    config = json.loads(STACKING_CONFIG_FILE.read_text(encoding="utf-8"))

    # Bases consolidadas e parciais (folds soltos de tentativas interrompidas).
    consolidadas = {p.stem for p in OOF_DIR.glob("*.npz") if "_fold" not in p.stem and "_amostra" not in p.stem}
    parciais = defaultdict(list)
    for p in sorted(OOF_DIR.glob("*_fold*.npz")):
        tag = re.sub(r"_fold\d+$", "", p.stem)
        if tag not in consolidadas and "_amostra" not in tag:
            parciais[tag].append(p)

    linhas = []
    for tag in sorted(consolidadas):
        dados = np.load(OOF_DIR / f"{tag}.npz")
        pred = np.array(CLASSES)[dados["oof"].argmax(1)]
        acc, recall, n_folds = metricas(pred, y, dados["fold_ids"])
        linhas.append({"modelo": tag, "descricao": DESCRICOES.get(tag, ""), "folds": n_folds,
                       "oof_accuracy": acc.mean(), "oof_std": acc.std(),
                       **{f"recall_{c}": recall[c] for c in CLASSES},
                       "no_modelo_final": tag in config["bases"]})

    for tag, arquivos in parciais.items():
        oof = np.full((len(y), len(CLASSES)), np.nan)
        fold_ids = np.full(len(y), -1)
        for f, p in enumerate(arquivos):
            d = np.load(p)
            oof[d["val_idx"]] = d["probs_val"]
            fold_ids[d["val_idx"]] = f
        pred = np.array(CLASSES)[np.nan_to_num(oof).argmax(1)]
        acc, recall, n_folds = metricas(pred, y, fold_ids)
        linhas.append({"modelo": f"{tag} (parcial)", "descricao": DESCRICOES.get(tag, ""), "folds": n_folds,
                       "oof_accuracy": acc.mean(), "oof_std": acc.std(),
                       **{f"recall_{c}": recall[c] for c in CLASSES},
                       "no_modelo_final": False})

    # Stacking final e referências nos testes.
    oof_pred = pd.read_csv(EP2_RESULTS_DIR / "stacking_oof_pred.csv")
    acc, recall, n_folds = metricas(oof_pred["y_pred"].to_numpy(), oof_pred["y_true"].to_numpy(),
                                    oof_pred["fold"].to_numpy())
    linhas.append({"modelo": "stacking (meta-LR, modelo final)",
                   "descricao": f"meta-LR sobre {' + '.join(config['bases'])} + lookup de duplicatas",
                   "folds": n_folds, "oof_accuracy": acc.mean(), "oof_std": acc.std(),
                   **{f"recall_{c}": recall[c] for c in CLASSES}, "no_modelo_final": True})

    mcnemar_path = EP2_RESULTS_DIR / "mcnemar.csv"
    if mcnemar_path.exists():
        m = pd.read_csv(mcnemar_path).iloc[0]
        for nome, acc_t in (("EP1 no teste 1", m["acc_ep1"]), ("EP2 no teste 1", m["acc_ep2"])):
            linhas.append({"modelo": nome, "descricao": f"McNemar exato p={m['mcnemar_exato_p']:.3f}",
                           "folds": None, "oof_accuracy": acc_t, "oof_std": None,
                           **{f"recall_{c}": None for c in CLASSES}, "no_modelo_final": None})

    tabela = pd.DataFrame(linhas).sort_values("oof_accuracy", ascending=False)
    salvar_tabela(tabela, EP2_RESULTS_DIR / "resumo_ep2.csv")
    with pd.option_context("display.width", 200):
        print(tabela.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
