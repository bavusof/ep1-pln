import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CURRENT_MODEL_FILE, EXPERIMENTS_SUMMARY_FILE


RESULT_COLUMNS = [
    "experiment_id",
    "experiment_name",
    "category",
    "representation",
    "model",
    "evaluation",
    "folds",
    "accuracy",
    "accuracy_std",
    "f1_macro",
    "f1_macro_std",
    "train_accuracy",
    "train_accuracy_std",
    "train_f1_macro",
    "train_f1_macro_std",
    "accuracy_gap",
    "selection_accuracy",
    "parameters",
    "notes",
    "timestamp_utc",
]


def _json_default(value: Any):
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo não serializável em JSON: {type(value).__name__}")


def serializar_parametros(parameters: Any) -> str:
    """Serializa parâmetros de forma estável para armazenamento no CSV."""
    if parameters is None:
        return ""
    return json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    )


def registrar_experimento(resultado: dict[str, Any]) -> None:
    """Adiciona ou substitui um experimento no resumo consolidado."""
    registro = {col: None for col in RESULT_COLUMNS}
    registro.update(resultado)
    registro["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    novo = pd.DataFrame([registro], columns=RESULT_COLUMNS)

    if EXPERIMENTS_SUMMARY_FILE.exists():
        atual = pd.read_csv(EXPERIMENTS_SUMMARY_FILE)
        if "experiment_id" in atual.columns:
            atual = atual[atual["experiment_id"] != registro["experiment_id"]]
    else:
        atual = pd.DataFrame(columns=RESULT_COLUMNS)

    df_resultados = pd.concat([atual, novo], ignore_index=True)
    EXPERIMENTS_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(EXPERIMENTS_SUMMARY_FILE, index=False)


def salvar_tabela(df: pd.DataFrame, caminho: Path) -> None:
    """Salva uma tabela de resultados garantindo a criação do diretório."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)


def salvar_modelo_atual(modelo: dict[str, Any]) -> None:
    """Salva a configuração e os resultados do modelo considerado atual."""
    CURRENT_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_MODEL_FILE.write_text(
        json.dumps(
            modelo,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
