"""Fine-tuning de um BERT pré-treinado em português para a tarefa de clareza.

Diferente dos experimentos com encoder congelado, aqui todos os pesos do
encoder são ajustados junto com a cabeça de classificação.
"""

import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import CLASSES, RANDOM_STATE


LABEL2ID = {classe: indice for indice, classe in enumerate(CLASSES)}
ID2LABEL = {indice: classe for classe, indice in LABEL2ID.items()}


def selecionar_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fixar_sementes(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenizar_head_tail(
    tokenizer,
    textos: Sequence[str],
    *,
    max_length: int,
    head_tokens: int,
) -> list[list[int]]:
    """Tokeniza mantendo o início e o fim de textos longos.

    Respostas longas costumam ter a informação principal no início e a
    conclusão/encaminhamento no fim, então o trecho central é descartado.
    """
    # Encoders (BERT) usam [CLS] ... [SEP]; decoders (Llama/Tucano) não têm
    # CLS/SEP: começam com <s> (BOS) e classificam pelo último token real.
    inicio = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    fim = tokenizer.sep_token_id
    prefixo = [inicio] if inicio is not None else []
    sufixo = [fim] if fim is not None else []
    corpo = max_length - len(prefixo) - len(sufixo)
    tail_tokens = corpo - head_tokens
    codificados = tokenizer(
        list(textos),
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]

    sequencias = []
    for ids in codificados:
        if len(ids) > corpo:
            ids = ids[:head_tokens] + (ids[-tail_tokens:] if tail_tokens > 0 else [])
        sequencias.append(prefixo + ids + sufixo)
    return sequencias


class _SequenciasDataset(Dataset):
    def __init__(self, sequencias, labels=None):
        self.sequencias = sequencias
        self.labels = labels

    def __len__(self):
        return len(self.sequencias)

    def __getitem__(self, indice):
        item = {"input_ids": self.sequencias[indice]}
        if self.labels is not None:
            item["labels"] = self.labels[indice]
        return item


def _criar_collate(pad_token_id: int):
    def collate(batch):
        tamanho = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), tamanho), pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(batch), tamanho), dtype=torch.long)
        for i, item in enumerate(batch):
            ids = item["input_ids"]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[i, : len(ids)] = 1
        saida = {"input_ids": input_ids, "attention_mask": attention}
        if "labels" in batch[0]:
            saida["labels"] = torch.tensor(
                [item["labels"] for item in batch], dtype=torch.long
            )
        return saida

    return collate


def _loader_ordenado(sequencias, batch_size, collate):
    """DataLoader de inferência ordenado por tamanho (menos padding)."""
    ordem = np.argsort([len(s) for s in sequencias])
    dataset = _SequenciasDataset([sequencias[i] for i in ordem])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return loader, ordem


@torch.no_grad()
def prever_probabilidades(
    model,
    tokenizer,
    textos: Sequence[str],
    *,
    max_length: int,
    head_tokens: int,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> np.ndarray:
    """Retorna probabilidades (n_textos, n_classes) na ordem de CLASSES."""
    device = device or selecionar_device()
    model.to(device).eval()
    sequencias = tokenizar_head_tail(
        tokenizer, textos, max_length=max_length, head_tokens=head_tokens
    )
    loader, ordem = _loader_ordenado(
        sequencias, batch_size, _criar_collate(tokenizer.pad_token_id)
    )

    partes = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        partes.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())

    probs_ordenadas = np.concatenate(partes)
    probs = np.empty_like(probs_ordenadas)
    probs[ordem] = probs_ordenadas
    return probs


def alvos_sord(alpha: float) -> torch.Tensor:
    """Alvos suaves ordinais (SORD; Díaz & Marathe, 2019).

    Linha i = distribuição-alvo da classe i: q_j ∝ exp(-alpha * |i - j|),
    com a ordem c1 < c234 < c5 dada por CLASSES.
    """
    posicoes = torch.arange(len(CLASSES), dtype=torch.float32)
    distancia = (posicoes[:, None] - posicoes[None, :]).abs()
    return torch.softmax(-alpha * distancia, dim=1)


def treinar_bert(
    textos_treino: Sequence[str],
    y_treino: Sequence[str],
    *,
    model_name: str,
    max_length: int = 256,
    head_tokens: int = 128,
    epochs: int = 3,
    batch_size: int = 16,
    grad_accum: int = 1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    textos_validacao: Sequence[str] | None = None,
    y_validacao: Sequence[str] | None = None,
    seed: int = RANDOM_STATE,
    log_every: int = 100,
    perda: str = "ce",
    alpha_sord: float = 2.0,
    lora: dict | None = None,
):
    """Faz fine-tuning e retorna (model, tokenizer, historico, probs_val_por_epoca).

    perda="ce": entropia cruzada padrão (one-hot).
    perda="sord": entropia cruzada contra alvos suaves ordinais (ver alvos_sord).
    lora={"r": 16, "alpha": 32, "dropout": 0.05}: congela o modelo e treina só
    adaptadores LoRA + cabeça de classificação (usado nos LLMs).
    """
    if perda not in {"ce", "sord"}:
        raise ValueError(f"perda desconhecida: {perda}")
    fixar_sementes(seed)
    device = selecionar_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        # Alguns checkpoints (ex.: Albertina) vêm em bfloat16; o treino é em float32.
        dtype=torch.float32,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if lora is not None:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type="SEQ_CLS",
                r=lora.get("r", 16),
                lora_alpha=lora.get("alpha", 32),
                lora_dropout=lora.get("dropout", 0.05),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        model.print_trainable_parameters()
    model = model.to(device)

    sequencias = tokenizar_head_tail(
        tokenizer, textos_treino, max_length=max_length, head_tokens=head_tokens
    )
    labels = [LABEL2ID[classe] for classe in y_treino]
    collate = _criar_collate(tokenizer.pad_token_id)
    gerador = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        _SequenciasDataset(sequencias, labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=gerador,
    )

    sem_decay = ("bias", "LayerNorm.weight")
    grupos = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(s in n for s in sem_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(s in n for s in sem_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grupos, lr=learning_rate)
    total_steps = epochs * math.ceil(len(loader) / grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    matriz_sord = alvos_sord(alpha_sord).to(device) if perda == "sord" else None

    historico = []
    probs_val_por_epoca = []
    for epoca in range(1, epochs + 1):
        model.train()
        inicio = time.time()
        perda_acumulada = 0.0
        acertos = 0
        vistos = 0
        for passo, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            if matriz_sord is None:
                saida = model(**batch)
                loss = saida.loss
            else:
                saida = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                log_probs = torch.log_softmax(saida.logits, dim=-1)
                loss = -(matriz_sord[batch["labels"]] * log_probs).sum(dim=-1).mean()
            (loss / grad_accum).backward()
            if passo % grad_accum == 0 or passo == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            perda_acumulada += loss.item() * len(batch["labels"])
            acertos += (saida.logits.argmax(-1) == batch["labels"]).sum().item()
            vistos += len(batch["labels"])
            if passo % log_every == 0:
                decorrido = time.time() - inicio
                print(
                    f"  época {epoca} passo {passo}/{len(loader)} "
                    f"loss={perda_acumulada / vistos:.4f} acc={acertos / vistos:.4f} "
                    f"({decorrido / passo:.2f}s/passo)",
                    flush=True,
                )

        registro = {
            "epoch": epoca,
            "train_loss": perda_acumulada / vistos,
            "train_accuracy_running": acertos / vistos,
            "epoch_seconds": time.time() - inicio,
        }

        if textos_validacao is not None:
            probs = prever_probabilidades(
                model,
                tokenizer,
                textos_validacao,
                max_length=max_length,
                head_tokens=head_tokens,
                device=device,
            )
            pred = np.array(CLASSES)[probs.argmax(1)]
            registro["val_accuracy"] = float((pred == np.asarray(y_validacao)).mean())
            probs_val_por_epoca.append(probs)

        historico.append(registro)
        print(f"Época {epoca}: {registro}", flush=True)

    return model, tokenizer, historico, probs_val_por_epoca


def salvar_modelo(model, tokenizer, diretorio: Path) -> None:
    diretorio.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(diretorio)
    tokenizer.save_pretrained(diretorio)


def carregar_modelo(diretorio: Path):
    tokenizer = AutoTokenizer.from_pretrained(diretorio)
    model = AutoModelForSequenceClassification.from_pretrained(diretorio)
    return model, tokenizer
