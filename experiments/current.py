"""Avalia e registra a configuração que está sendo tratada como modelo atual.

Para testar uma nova configuração, edite apenas CURRENT_MODEL abaixo.
Depois execute:
    python -m experiments.current
"""

from src.config import CURRENT_MODEL_FILE, N_SPLITS
from src.data import carregar_dados_treino
from src.evaluation import avaliar_modelo
from src.models import criar_pipeline
from src.results import registrar_experimento, salvar_modelo_atual


CURRENT_MODEL = {
    "experiment_id": "current_model",
    "name": "tfidf_word_current",
    "description": "Configuração atual em desenvolvimento.",
    "type": "tfidf_word",
    "representation": "TF-IDF palavra",
    "model": "Regressão Logística",
    "parameters": {
        "ngram_range": (1, 2),
        "min_df": 1,
        "max_df": 1.0,
        "sublinear_tf": True,
        "C": 1.0,
    },
}


def executar_modelo_atual() -> dict:
    X, y = carregar_dados_treino()
    pipeline = criar_pipeline(
        {
            "tipo": CURRENT_MODEL["type"],
            "parametros": CURRENT_MODEL["parameters"],
        }
    )

    _, resumo, resultados_folds = avaliar_modelo(
        pipeline,
        X,
        y,
        n_splits=N_SPLITS,
    )

    print("\n==============================")
    print("MODELO ATUAL")
    print("==============================")
    print(f"Nome: {CURRENT_MODEL['name']}")
    print(f"Descrição: {CURRENT_MODEL['description']}")
    print("\nParâmetros:")
    print(CURRENT_MODEL["parameters"])
    print("\nResultados por fold:")
    print(resultados_folds.to_string(index=False))
    print("\nResumo:")
    print(f"Accuracy: {resumo['accuracy']:.4f} ± {resumo['accuracy_std']:.4f}")
    print(f"F1 Macro: {resumo['f1_macro']:.4f} ± {resumo['f1_macro_std']:.4f}")
    print(f"Accuracy treino: {resumo['train_accuracy']:.4f}")
    print(f"Gap: {resumo['accuracy_gap']:.4f}")

    resultado_registro = {
        "experiment_id": CURRENT_MODEL["experiment_id"],
        "experiment_name": CURRENT_MODEL["name"],
        "category": "current",
        "representation": CURRENT_MODEL["representation"],
        "model": CURRENT_MODEL["model"],
        "evaluation": "cross_validation",
        "folds": N_SPLITS,
        **resumo,
        "parameters": CURRENT_MODEL["parameters"],
        "notes": CURRENT_MODEL["description"],
    }
    registrar_experimento(resultado_registro)

    modelo_atual = {
        **CURRENT_MODEL,
        "evaluation": {
            "method": "StratifiedKFold",
            "folds": N_SPLITS,
            **resumo,
        },
    }
    salvar_modelo_atual(modelo_atual)

    print(f"\nModelo atual salvo em: {CURRENT_MODEL_FILE}")
    return modelo_atual


def main() -> None:
    executar_modelo_atual()


if __name__ == "__main__":
    main()
