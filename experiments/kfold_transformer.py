"""Fine-tuning de um transformer em 5 folds para o stacking do EP2.

Para cada fold: treina no restante do treino, prevê o fold (OOF) e os
testes 1 e 2. Cada fold é salvo assim que termina e é pulado em uma nova
execução (os treinos são longos). Os pesos do modelo não são salvos.

    python -m experiments.kfold_transformer --model neuralmind/bert-base-portuguese-cased --tag bertimbau_base
    python -m experiments.kfold_transformer ... --amostra 300 --epochs 1   # teste rápido

Use `caffeinate -i` no macOS para o sistema não dormir durante o treino.
"""

import argparse
import gc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.config import CLASSES, FINETUNE_PARAMS, N_SPLITS
from src.data import carregar_dados_treino
from src.finetune_bert import prever_probabilidades, selecionar_device, treinar_bert
from src.oof import acuracia_por_fold, arquivo_fold, carregar_textos_teste, consolidar, folds_ep2, salvar_fold
from src.results import registrar_experimento
from src.text_preprocessing import limpar_para_bert
from src.ruido_rotulos import mascara_manter
from src.tratamento_dados import tratar_treino


def liberar_memoria() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=FINETUNE_PARAMS["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=FINETUNE_PARAMS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=FINETUNE_PARAMS["max_length"])
    parser.add_argument("--head-tokens", type=int, default=FINETUNE_PARAMS["head_tokens"])
    parser.add_argument("--amostra", type=int, default=None, help="usa só N linhas (teste rápido)")
    parser.add_argument("--perda", choices=["ce", "sord"], default="ce")
    parser.add_argument("--alpha-sord", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true", help="treina só adaptadores LoRA (LLMs)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--cleanlab-frac", type=float, default=None,
                        help="remove essa fração de linhas de menor qualidade de rótulo (confident learning)")
    parser.add_argument("--treino-limpo", action="store_true",
                        help="aplica src/tratamento_dados.py à parte de treino de cada fold")
    args = parser.parse_args()
    if args.cleanlab_frac and args.treino_limpo:
        parser.error("--cleanlab-frac e --treino-limpo não são combináveis")

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
    params = {
        "max_length": args.max_length,
        "head_tokens": args.head_tokens,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": FINETUNE_PARAMS["weight_decay"],
        "warmup_ratio": FINETUNE_PARAMS["warmup_ratio"],
        "perda": args.perda,
        "alpha_sord": args.alpha_sord,
        "seed": args.seed,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": 0.05} if args.lora else None,
    }
    print(f"Modelo: {args.model} | tag: {tag} | device: {selecionar_device()} | params: {params}")

    for fold, (treino, validacao) in enumerate(splits):
        if arquivo_fold(tag, fold).exists():
            print(f"Fold {fold}: já concluído, pulando.")
            continue
        print(f"\n===== Fold {fold + 1}/{len(splits)} (treino={len(treino)}, validação={len(validacao)}) =====", flush=True)
        textos_treino, rotulos_treino = X_limpo[treino], y_arr[treino]
        if args.treino_limpo:
            originais, rotulos_treino, relatorio = tratar_treino(X.to_numpy(dtype=object)[treino], y_arr[treino])
            textos_treino = np.array([limpar_para_bert(t) for t in originais], dtype=object)
            print(f"Treino limpo: {relatorio}", flush=True)
        if args.cleanlab_frac:
            manter, rel = mascara_manter(X.to_numpy(dtype=object)[treino], y_arr[treino], args.cleanlab_frac, fold=fold)
            textos_treino, rotulos_treino = textos_treino[manter], rotulos_treino[manter]
            print(f"Cleanlab: {rel}", flush=True)
        model, tokenizer, historico, probs_por_epoca = treinar_bert(
            textos_treino.tolist(),
            rotulos_treino.tolist(),
            model_name=args.model,
            textos_validacao=X_limpo[validacao].tolist(),
            y_validacao=y_arr[validacao].tolist(),
            **params,
        )
        # Época fixa (a última): escolher a melhor época pela validação deixaria o OOF otimista.
        probs_val = probs_por_epoca[-1]
        prever = lambda textos: prever_probabilidades(
            model, tokenizer, textos, max_length=args.max_length, head_tokens=args.head_tokens
        )
        salvar_fold(
            tag,
            fold,
            val_idx=validacao,
            probs_val=probs_val,
            probs_test1=prever(testes_limpos["test1"]),
            probs_test2=prever(testes_limpos["test2"]),
        )
        print(f"Fold {fold}: acurácia validação = {historico[-1]['val_accuracy']:.4f}", flush=True)
        del model, tokenizer
        liberar_memoria()

    base = consolidar(tag, len(y_arr), len(splits))
    acc_folds = acuracia_por_fold(base["oof"], base["fold_ids"], y_arr)
    pred = np.array(CLASSES)[base["oof"].argmax(1)]
    print(f"\n{tag}: acurácia OOF = {acc_folds.mean():.4f} ± {acc_folds.std():.4f} | por fold: {np.round(acc_folds, 4)}")

    if not args.amostra:
        registrar_experimento(
            {
                "experiment_id": f"ep2_oof_{tag}",
                "experiment_name": f"EP2 base {tag} (5-fold OOF)",
                "category": "ep2_base",
                "representation": "transformer fine-tuned",
                "model": args.model,
                "evaluation": "5-fold StratifiedGroupKFold por texto (OOF)",
                "folds": N_SPLITS,
                "accuracy": float(acc_folds.mean()),
                "accuracy_std": float(acc_folds.std()),
                "f1_macro": float(f1_score(y_arr, pred, average="macro")),
                "parameters": {**params, "treino_limpo": args.treino_limpo, "cleanlab_frac": args.cleanlab_frac},
                "notes": "Época final fixa; previsões de teste = média dos 5 modelos dos folds.",
            }
        )


if __name__ == "__main__":
    main()
