from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def carregar_modelo_token_bert(
    model_name: str,
    device: str | None = None,
):
    """
    Carrega o Sentence Transformer e acessa o Transformer interno.

    O encoder fica congelado. Nenhum treinamento é realizado aqui.
    """

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    transformer_module = None

    for module in model:
        if (
            hasattr(module, "auto_model")
            and hasattr(module, "tokenizer")
        ):
            transformer_module = module
            break

    if transformer_module is None:
        raise RuntimeError(
            "Não foi possível localizar o módulo Transformer "
            "interno do Sentence Transformer."
        )

    tokenizer = transformer_module.tokenizer
    transformer = transformer_module.auto_model

    transformer.eval()

    for parameter in transformer.parameters():
        parameter.requires_grad = False

    return (
        model,
        tokenizer,
        transformer,
    )


def tokenizar_documentos(
    textos: Iterable[str],
    tokenizer,
    max_length: int,
):
    """
    Tokeniza os documentos usando o tokenizer do BERT.

    O próprio tokenizer adiciona os tokens especiais
    ([CLS], [SEP], etc.) e realiza o truncamento.

    Retorna:
        token_lists:
            subwords sem tokens especiais;

        input_ids:
            ids completos, incluindo tokens especiais;

        attention_masks:
            máscaras completas.
    """

    textos = [
        "" if texto is None else str(texto)
        for texto in textos
    ]

    token_lists = []
    input_ids = []
    attention_masks = []

    for texto in textos:

        encoded = tokenizer(
            texto,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

        ids = encoded["input_ids"]

        attention_mask = encoded[
            "attention_mask"
        ]

        special_mask = encoded[
            "special_tokens_mask"
        ]

        tokens_completos = (
            tokenizer.convert_ids_to_tokens(ids)
        )

        tokens = [
            token
            for token, is_special in zip(
                tokens_completos,
                special_mask,
            )
            if not is_special
        ]

        token_lists.append(
            tokens
        )

        input_ids.append(
            ids
        )

        attention_masks.append(
            attention_mask
        )

    return (
        token_lists,
        input_ids,
        attention_masks,
    )


def _criar_pesos_token_documento(
    tokens: list[str],
    tfidf_vectorizer,
) -> np.ndarray:
    """
    Produz um peso TF-IDF para cada ocorrência de subword.

    O TF-IDF é calculado pelo próprio TfidfVectorizer,
    garantindo que IDF, sublinear_tf e normalização sejam
    tratados pela implementação do sklearn.
    """

    if not tokens:
        return np.zeros(
            0,
            dtype=np.float32,
        )

    # Mantém a estrutura de entrada esperada pelo vectorizer.
    contagem = Counter(tokens)

    if not contagem:
        return np.zeros(
            len(tokens),
            dtype=np.float32,
        )

    linha = (
        tfidf_vectorizer
        .transform([tokens])
        .tocsr()[0]
    )

    pesos_por_coluna = dict(
        zip(
            linha.indices,
            linha.data,
        )
    )

    pesos = np.array(
        [
            pesos_por_coluna.get(
                tfidf_vectorizer.vocabulary_.get(
                    token
                ),
                0.0,
            )
            for token in tokens
        ],
        dtype=np.float32,
    )

    return pesos


def gerar_embeddings_token_weighted(
    token_lists: list[list[str]],
    input_ids: list[list[int]],
    attention_masks: list[list[int]],
    tfidf_vectorizer,
    transformer,
    tokenizer,
    *,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """
    Gera embeddings documentais através de:

        embedding_documento =
            soma(TFIDF(token) * embedding(token))
            -------------------------------------
                  soma(TFIDF(token))

    O embedding de cada token é contextual.
    """

    if device is None:

        try:
            device = next(
                transformer.parameters()
            ).device

        except StopIteration:
            device = torch.device(
                "cpu"
            )

    else:
        device = torch.device(
            device
        )

    hidden_size = int(
        transformer.config.hidden_size
    )

    embeddings = np.zeros(
        (
            len(input_ids),
            hidden_size,
        ),
        dtype=np.float32,
    )

    special_ids = set(
        tokenizer.all_special_ids
    )

    transformer.eval()

    with torch.no_grad():

        for inicio in range(
            0,
            len(input_ids),
            batch_size,
        ):
            fim = min(
                inicio + batch_size,
                len(input_ids),
            )

            batch_records = [
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                }
                for ids, mask in zip(
                    input_ids[inicio:fim],
                    attention_masks[inicio:fim],
                )
            ]

            batch = tokenizer.pad(
                batch_records,
                padding=True,
                return_tensors="pt",
            )

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = transformer(
                **batch
            )

            hidden = (
                outputs.last_hidden_state
                .float()
            )

            pesos = torch.zeros(
                hidden.shape[:2],
                dtype=torch.float32,
                device=device,
            )

            for local_idx, tokens in enumerate(
                token_lists[inicio:fim]
            ):

                ids = input_ids[
                    inicio + local_idx
                ]

                token_weights = (
                    _criar_pesos_token_documento(
                        tokens,
                        tfidf_vectorizer,
                    )
                )

                cursor = 0

                for pos, token_id in enumerate(
                    ids
                ):

                    if token_id in special_ids:
                        continue

                    if cursor >= len(
                        token_weights
                    ):
                        break

                    pesos[
                        local_idx,
                        pos,
                    ] = float(
                        token_weights[
                            cursor
                        ]
                    )

                    cursor += 1

            soma_pesos = pesos.sum(
                dim=1,
                keepdim=True,
            )

            pooled = (
                hidden
                * pesos.unsqueeze(-1)
            ).sum(
                dim=1
            )

            pooled = (
                pooled
                / soma_pesos.clamp_min(
                    1e-8
                )
            )

            # Normalização L2 do embedding documental.
            norma = torch.linalg.vector_norm(
                pooled,
                dim=1,
                keepdim=True,
            )

            pooled = (
                pooled
                / norma.clamp_min(
                    1e-8
                )
            )

            embeddings[
                inicio:fim
            ] = pooled.cpu().numpy()

    return embeddings