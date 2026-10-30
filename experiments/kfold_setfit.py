"""SetFit em 5 folds para o stacking do EP2.

SetFit = (1) fine-tuning contrastivo de um Sentence Transformer com pares
de respostas (mesma classe aproxima, classes diferentes afasta) e
(2) Regressão Logística sobre os embeddings ajustados.

Mesmo protocolo de experiments/kfold_transformer.py: folds agrupados por
texto, OOF no treino, média dos 5 modelos nos testes, retomável por fold.

    python -m experiments.kfold_setfit --body rufimelo/bert-large-portuguese-cased-sts --tag setfit_bertlarge_sts
    python -m experiments.kfold_setfit ... --amostra 300 --max-steps 50   # teste rápido
"""

import argparse
import gc
import tempfile

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score

from src.config import CLASSES, N_SPLITS, RANDOM_STATE
from src.data import carregar_dados_treino
from src.oof import acuracia_por_fold, arquivo_fold, carregar_textos_teste, consolidar, folds_ep2, salvar_fold
from src.results import registrar_experimento
from src.text_preprocessing import limpar_para_bert


def liberar_memoria() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def probs_na_ordem(model, textos) -> np.ndarray:
    probs = np.asarray(model.predict_proba(list(textos), as_numpy=True))
    classes = list(model.model_head.classes_)
    return probs[:, [classes.index(c) for c in CLASSES]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)  # 4 pares = 8 sequências; 8 pares estoura os 24 GB
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--amostra", type=int, default=None)
    args = parser.parse_args()

    X, y = carregar_dados_treino()
    testes = carregar_textos_teste()
    tag = args.tag
    if args.amostra:
        amostra = pd.DataFrame({"x": X, "y": y}).sample(args.amostra, random_state=0).reset_index(drop=True)
        X, y = amostra["x"], amostra["y"]
        testes = {k: v.head(20) for k, v in testes.items()}
        tag = f"{tag}_amostra"

    X_limpo = X.map(limpar_para_bert).to_numpy(dtype=object)
    y_arr = y.to_numpy()
    testes_limpos = {k: v.map(limpar_para_bert).tolist() for k, v in testes.items()}
    splits = folds_ep2(X, y)
    params = {"max_steps": args.max_steps, "batch_size": args.batch_size, "body_learning_rate": args.lr,
              "max_length": args.max_length, "num_iterations": 1}
    print(f"SetFit | corpo: {args.body} | tag: {tag} | params: {params}", flush=True)

    for fold, (treino, validacao) in enumerate(splits):
        if arquivo_fold(tag, fold).exists():
            print(f"Fold {fold}: já concluído, pulando.")
            continue
        print(f"\n===== Fold {fold + 1}/{len(splits)} (treino={len(treino)}, validação={len(validacao)}) =====", flush=True)
        model = SetFitModel.from_pretrained(args.body)
        # O TrainingArguments.max_length não é repassado ao corpo; sem isso o
        # modelo usa 512 tokens e o large estoura a memória da GPU.
        model.model_body.max_seq_length = args.max_length
        with tempfile.TemporaryDirectory() as saida:
            treino_args = TrainingArguments(
                output_dir=saida,
                batch_size=args.batch_size,
                num_epochs=1,
                max_steps=args.max_steps,
                # num_iterations sorteia 2 pares por texto (em vez de todas as combinações).
                num_iterations=1,
                body_learning_rate=args.lr,
                max_length=args.max_length,
                seed=RANDOM_STATE,
                save_strategy="no",
                logging_steps=100,
                report_to="none",
            )
            dados = Dataset.from_dict({"text": X_limpo[treino].tolist(), "label": y_arr[treino].tolist()})
            Trainer(model=model, args=treino_args, train_dataset=dados).train()

        probs_val = probs_na_ordem(model, X_limpo[validacao])
        acc = (np.array(CLASSES)[probs_val.argmax(1)] == y_arr[validacao]).mean()
        salvar_fold(
            tag,
            fold,
            val_idx=validacao,
            probs_val=probs_val,
            probs_test1=probs_na_ordem(model, testes_limpos["test1"]),
            probs_test2=probs_na_ordem(model, testes_limpos["test2"]),
        )
        print(f"Fold {fold}: acurácia validação = {acc:.4f}", flush=True)
        del model
        liberar_memoria()

    base = consolidar(tag, len(y_arr), len(splits))
    acc_folds = acuracia_por_fold(base["oof"], base["fold_ids"], y_arr)
    pred = np.array(CLASSES)[base["oof"].argmax(1)]
    print(f"\n{tag}: acurácia OOF = {acc_folds.mean():.4f} ± {acc_folds.std():.4f} | por fold: {np.round(acc_folds, 4)}")

    if not args.amostra:
        registrar_experimento(
            {
                "experiment_id": f"ep2_oof_{tag}",
                "experiment_name": f"EP2 base {tag} (SetFit, 5-fold OOF)",
                "category": "ep2_base",
                "representation": "Sentence Transformer ajustado por aprendizado contrastivo (SetFit)",
                "model": f"SetFit ({args.body}) + Regressão Logística",
                "evaluation": "5-fold StratifiedGroupKFold por texto (OOF)",
                "folds": N_SPLITS,
                "accuracy": float(acc_folds.mean()),
                "accuracy_std": float(acc_folds.std()),
                "f1_macro": float(f1_score(y_arr, pred, average="macro")),
                "parameters": {"body": args.body, **params},
                "notes": "Previsões de teste = média dos 5 modelos dos folds.",
            }
        )


if __name__ == "__main__":
    main()
