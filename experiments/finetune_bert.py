"""Fine-tuning do BERTimbau para classificação de clareza.

Modos:
    python -m experiments.finetune_bert holdout
        Treina em 80% e avalia no mesmo holdout estratificado (20%,
        random_state=42) usado pelo baseline oficial. Salva as probabilidades
        do holdout por época para o ensemble.

    python -m experiments.finetune_bert full [--epochs N]
        Treina no conjunto de treino completo e salva o modelo em models/
        para a predição do conjunto de teste.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSES,
    FINETUNE_FULL_MODEL_DIR,
    FINETUNE_HISTORY_FILE,
    FINETUNE_HOLDOUT_PROBS_FILE,
    FINETUNE_MODEL_NAME,
    FINETUNE_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data import carregar_dados_treino
from src.finetune_bert import salvar_modelo, treinar_bert
from src.results import registrar_experimento, salvar_tabela


def dividir_holdout(X, y):
    """Mesmo split estratificado do baseline oficial em holdout."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def executar_holdout(params: dict) -> None:
    X, y = carregar_dados_treino()
    X_train, X_val, y_train, y_val = dividir_holdout(X, y)

    _, _, historico, probs_por_epoca = treinar_bert(
        X_train.tolist(),
        y_train.tolist(),
        model_name=FINETUNE_MODEL_NAME,
        textos_validacao=X_val.tolist(),
        y_validacao=y_val.tolist(),
        **params,
    )

    df_hist = pd.DataFrame(historico)
    salvar_tabela(df_hist, FINETUNE_HISTORY_FILE)

    linhas = []
    for epoca, probs in enumerate(probs_por_epoca, start=1):
        bloco = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES])
        bloco.insert(0, "index", X_val.index.to_numpy())
        bloco.insert(1, "epoch", epoca)
        bloco["y_true"] = y_val.to_numpy()
        linhas.append(bloco)
    salvar_tabela(pd.concat(linhas, ignore_index=True), FINETUNE_HOLDOUT_PROBS_FILE)

    # Textos do holdout que também aparecem (idênticos) no treino.
    vistos = set(X_train)
    mascara_inedito = ~X_val.isin(vistos).to_numpy()

    melhor = int(df_hist["val_accuracy"].idxmax())
    probs = probs_por_epoca[melhor]
    pred = np.array(CLASSES)[probs.argmax(1)]
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, average="macro")
    acc_inedito = accuracy_score(y_val[mascara_inedito], pred[mascara_inedito])

    print("\n==============================")
    print("FINE-TUNING BERT - HOLDOUT")
    print("==============================")
    print(df_hist.to_string(index=False))
    print(f"Melhor época: {melhor + 1}")
    print(f"Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")
    print(f"Accuracy em textos inéditos ({mascara_inedito.sum()}): {acc_inedito:.4f}")

    registrar_experimento(
        {
            "experiment_id": "finetune_bert_holdout",
            "experiment_name": "Fine-tuning BERTimbau",
            "category": "fine_tuning",
            "representation": "BERT fine-tuned (head+tail truncation)",
            "model": FINETUNE_MODEL_NAME,
            "evaluation": "holdout",
            "accuracy": acc,
            "f1_macro": f1,
            "train_accuracy": df_hist.loc[melhor, "train_accuracy_running"],
            "accuracy_gap": df_hist.loc[melhor, "train_accuracy_running"] - acc,
            "parameters": {**params, "best_epoch": melhor + 1},
            "notes": (
                "Encoder ajustado de ponta a ponta. Holdout estratificado de 20% "
                "(mesmo split do baseline oficial em holdout). "
                f"Accuracy em textos do holdout ausentes do treino: {acc_inedito:.4f}. "
                "train_accuracy é a média corrente durante a época."
            ),
        }
    )


def executar_full(params: dict) -> None:
    X, y = carregar_dados_treino()
    model, tokenizer, historico, _ = treinar_bert(
        X.tolist(),
        y.tolist(),
        model_name=FINETUNE_MODEL_NAME,
        **params,
    )
    salvar_modelo(model, tokenizer, FINETUNE_FULL_MODEL_DIR)
    print(pd.DataFrame(historico).to_string(index=False))
    print(f"\nModelo salvo em: {FINETUNE_FULL_MODEL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("modo", choices=["holdout", "full"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    params = dict(FINETUNE_PARAMS)
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.max_length is not None:
        params["max_length"] = args.max_length

    if args.modo == "holdout":
        executar_holdout(params)
    else:
        executar_full(params)


if __name__ == "__main__":
    main()
