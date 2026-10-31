"""Média das probabilidades de várias sementes de um mesmo modelo em uma base nova.

As sementes precisam ter usado os mesmos folds (mesmos fold_ids).

    python -m experiments.media_sementes --saida bertimbau_base_sord_3seeds \
        bertimbau_base_sord_s42 bertimbau_base_sord_s7 bertimbau_base_sord_s123
"""

import argparse

import numpy as np

from src.config import CLASSES, N_SPLITS
from src.data import carregar_dados_treino
from src.oof import acuracia_por_fold, arquivo_base, carregar_base
from src.results import registrar_experimento


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saida", required=True)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()

    bases = [carregar_base(tag) for tag in args.tags]
    fold_ids = bases[0]["fold_ids"]
    for tag, base in zip(args.tags, bases):
        assert (base["fold_ids"] == fold_ids).all(), f"{tag} usa folds diferentes"

    media = {chave: np.mean([b[chave] for b in bases], axis=0) for chave in ("oof", "test1", "test2")}
    np.savez(arquivo_base(args.saida), fold_ids=fold_ids, **media)

    _, y = carregar_dados_treino()
    y_arr = y.to_numpy()
    for tag, base in zip(args.tags, bases):
        acc = acuracia_por_fold(base["oof"], fold_ids, y_arr)
        print(f"{tag}: {acc.mean():.4f} ± {acc.std():.4f}")
    acc = acuracia_por_fold(media["oof"], fold_ids, y_arr)
    pred = np.array(CLASSES)[media["oof"].argmax(1)]
    recall = {c: float((pred[y_arr == c] == c).mean()) for c in CLASSES}
    print(f"{args.saida} (média): {acc.mean():.4f} ± {acc.std():.4f} | recall {recall}")

    registrar_experimento(
        {
            "experiment_id": f"ep2_oof_{args.saida}",
            "experiment_name": f"EP2 base {args.saida} (média de {len(args.tags)} sementes)",
            "category": "ep2_base",
            "representation": "média de probabilidades",
            "model": " + ".join(args.tags),
            "evaluation": "5-fold StratifiedGroupKFold por texto (OOF)",
            "folds": N_SPLITS,
            "accuracy": float(acc.mean()),
            "accuracy_std": float(acc.std()),
        }
    )


if __name__ == "__main__":
    main()
