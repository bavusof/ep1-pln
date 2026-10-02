from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


def carregar_ou_gerar_embeddings(
    textos: Iterable[str],
    *,
    model_name: str,
    output_file: Path,
    batch_size: int = 16,
    normalize_embeddings: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """
    Gera embeddings com um Sentence Transformer pré-treinado.

    O modelo é usado apenas como extrator fixo de características:
    não há treinamento nem ajuste nos rótulos do projeto. Por isso,
    os embeddings podem ser calculados uma única vez antes da CV.

    Se output_file existir, os embeddings são carregados do cache.
    """
    textos = [
        "" if texto is None else str(texto)
        for texto in textos
    ]

    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_file = output_file.with_suffix(".json")

    if output_file.exists() and metadata_file.exists():
        try:
            metadata = metadata_file.read_text(
                encoding="utf-8"
            )

            esperado = (
                f'"model_name": "{model_name}"'
            )

            if esperado in metadata:
                embeddings = np.load(
                    output_file
                )

                if embeddings.shape[0] == len(textos):
                    return embeddings.astype(
                        np.float32,
                        copy=False,
                    )
        except Exception:
            # Cache inválido: gera novamente.
            pass

    print("\n==============================")
    print("SENTENCE TRANSFORMER")
    print("==============================")
    print(f"Modelo: {model_name}")
    print(f"Documentos: {len(textos)}")
    print(f"Batch size: {batch_size}")

    if device:
        print(f"Device: {device}")
    else:
        print("Device: automático")

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    embeddings = model.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    np.save(
        output_file,
        embeddings,
    )

    metadata_file.write_text(
        (
            "{\n"
            f'  "model_name": "{model_name}",\n'
            f'  "batch_size": {batch_size},\n'
            f'  "normalize_embeddings": '
            f'{str(normalize_embeddings).lower()},\n'
            f'  "n_documents": {len(textos)},\n'
            f'  "dimension": {embeddings.shape[1]}\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    return embeddings
