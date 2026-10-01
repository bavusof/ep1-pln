import argparse

import pandas as pd

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    RANDOM_STATE,
    TUNING_CHAR_FILE,
    TUNING_WORD_CHAR_FILE,
    TUNING_WORD_FILE,
)
from src.data import carregar_dados_treino
from src.evaluation import criar_grid_search, avaliar_modelo
from src.models import (
    criar_pipeline_tfidf_caracteres,
    criar_pipeline_tfidf_word,
    criar_pipeline_tfidf_word_char,
)
from src.results import registrar_experimento, salvar_tabela
from sklearn.model_selection import StratifiedKFold


EXPERIMENTS = {
    "tfidf_word": {
        "name": "TF-IDF palavra",
        "representation": "TF-IDF palavra",
        "output": TUNING_WORD_FILE,
        "pipeline_factory": criar_pipeline_tfidf_word,
        "param_grid": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2, 5],
            "tfidf__sublinear_tf": [False, True],
            "tfidf__max_df": [1.0],
            "logistic__C": [1.0],
        },
        "result_columns": [
            "param_tfidf__ngram_range",
            "param_tfidf__min_df",
            "param_tfidf__sublinear_tf",
            "param_tfidf__max_df",
            "param_logistic__C",
            "mean_train_accuracy",
            "std_train_accuracy",
            "mean_test_accuracy",
            "std_test_accuracy",
        ],
        "rename": {
            "param_tfidf__ngram_range": "ngram_range",
            "param_tfidf__min_df": "min_df",
            "param_tfidf__sublinear_tf": "sublinear_tf",
            "param_tfidf__max_df": "max_df",
            "param_logistic__C": "C",
            "mean_train_accuracy": "train_accuracy_3fold",
            "std_train_accuracy": "train_accuracy_std_3fold",
            "mean_test_accuracy": "accuracy_3fold",
            "std_test_accuracy": "accuracy_std_3fold",
        },
        "make_pipeline": lambda: criar_pipeline_tfidf_word(),
    },
    "tfidf_char": {
        "name": "TF-IDF caracteres",
        "representation": "TF-IDF caracteres",
        "output": TUNING_CHAR_FILE,
        "pipeline_factory": criar_pipeline_tfidf_caracteres,
        "param_grid": [
            {
                "tfidf__analyzer": ["char"],
                "tfidf__ngram_range": [(3, 5), (4, 6)],
                "tfidf__min_df": [2],
                "tfidf__sublinear_tf": [True],
                "logistic__C": [0.5, 1.0, 2.0],
            },
            {
                "tfidf__analyzer": ["char_wb"],
                "tfidf__ngram_range": [(3, 5), (4, 6)],
                "tfidf__min_df": [2],
                "tfidf__sublinear_tf": [True],
                "logistic__C": [0.5, 1.0, 2.0],
            },
        ],
        "result_columns": [
            "param_tfidf__analyzer",
            "param_tfidf__ngram_range",
            "param_tfidf__min_df",
            "param_tfidf__sublinear_tf",
            "param_logistic__C",
            "mean_train_accuracy",
            "std_train_accuracy",
            "mean_test_accuracy",
            "std_test_accuracy",
        ],
        "rename": {
            "param_tfidf__analyzer": "analyzer",
            "param_tfidf__ngram_range": "ngram_range",
            "param_tfidf__min_df": "min_df",
            "param_tfidf__sublinear_tf": "sublinear_tf",
            "param_logistic__C": "C",
            "mean_train_accuracy": "train_accuracy_3fold",
            "std_train_accuracy": "train_accuracy_std_3fold",
            "mean_test_accuracy": "accuracy_3fold",
            "std_test_accuracy": "accuracy_std_3fold",
        },
        "make_pipeline": lambda: criar_pipeline_tfidf_caracteres(),
    },
    "tfidf_word_char": {
        "name": "TF-IDF palavra + caractere",
        "representation": "TF-IDF palavra + caractere",
        "output": TUNING_WORD_CHAR_FILE,
        "pipeline_factory": criar_pipeline_tfidf_word_char,
        "param_grid": {
            "features__word__ngram_range": [(1, 1), (1, 2)],
            "features__char__ngram_range": [(3, 5), (4, 6)],
            "logistic__C": [0.5, 1.0],
        },
        "result_columns": [
            "param_features__word__ngram_range",
            "param_features__char__ngram_range",
            "param_logistic__C",
            "mean_train_accuracy",
            "std_train_accuracy",
            "mean_test_accuracy",
            "std_test_accuracy",
        ],
        "rename": {
            "param_features__word__ngram_range": "word_ngram_range",
            "param_features__char__ngram_range": "char_ngram_range",
            "param_logistic__C": "C",
            "mean_train_accuracy": "train_accuracy_3fold",
            "std_train_accuracy": "train_accuracy_std_3fold",
            "mean_test_accuracy": "accuracy_3fold",
            "std_test_accuracy": "accuracy_std_3fold",
        },
        "make_pipeline": lambda: criar_pipeline_tfidf_word_char(),
    },
}


def _normalizar_parametros_grid(params: dict) -> dict:
    """Converte valores do GridSearchCV em valores serializáveis."""
    normalizados = {}
    for chave, valor in params.items():
        if isinstance(valor, tuple):
            normalizados[chave] = tuple(int(item) for item in valor)
        elif hasattr(valor, "item"):
            normalizados[chave] = valor.item()
        else:
            normalizados[chave] = valor
    return normalizados


def preparar_resultados_grid(grid, experimento: dict) -> pd.DataFrame:
    resultados = pd.DataFrame(grid.cv_results_)[experimento["result_columns"]].copy()
    resultados.rename(columns=experimento["rename"], inplace=True)
    resultados["accuracy_gap_3fold"] = (
        resultados["train_accuracy_3fold"]
        - resultados["accuracy_3fold"]
    )
    resultados.sort_values("accuracy_3fold", ascending=False, inplace=True)
    resultados.reset_index(drop=True, inplace=True)
    return resultados


def executar_tuning(nome: str) -> tuple[dict, pd.DataFrame]:
    if nome not in EXPERIMENTS:
        raise ValueError(
            f"Experimento desconhecido: {nome}. "
            f"Opções: {', '.join(EXPERIMENTS)}"
        )

    config = EXPERIMENTS[nome]
    X, y = carregar_dados_treino()

    print("\n==============================")
    print(f"TUNING - {config['name'].upper()}")
    print("==============================")
    print(f"Folds de seleção: {N_SPLITS_EXPLORATORIA}")

    pipeline = config["make_pipeline"]()

    cv_exploratoria = StratifiedKFold(
        n_splits=N_SPLITS_EXPLORATORIA,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid = criar_grid_search(
        pipeline,
        config["param_grid"],
        cv=cv_exploratoria,
        n_jobs=2,
    )
    grid.fit(X, y)

    resultados = preparar_resultados_grid(grid, config)

    print("\nTop configurações da exploração:")
    print(resultados.to_string(index=False))

    _, confirmacao, resultados_folds = avaliar_modelo(
        grid.best_estimator_,
        X,
        y,
        n_splits=N_SPLITS,
    )

    melhor_config = _normalizar_parametros_grid(grid.best_params_)

    print("\n==============================")
    print("CONFIRMAÇÃO - 5-FOLD CV")
    print("==============================")
    print("Parâmetros:", melhor_config)
    print(
        f"Accuracy: {confirmacao['accuracy']:.4f} "
        f"± {confirmacao['accuracy_std']:.4f}"
    )
    print(
        f"F1 Macro: {confirmacao['f1_macro']:.4f} "
        f"± {confirmacao['f1_macro_std']:.4f}"
    )
    print(f"Accuracy treino: {confirmacao['train_accuracy']:.4f}")
    print(f"Gap: {confirmacao['accuracy_gap']:.4f}")

    resultado = {
        "experiment_id": f"tuning_{nome}",
        "experiment_name": config["name"],
        "category": "tuning",
        "representation": config["representation"],
        "model": "Regressão Logística",
        "evaluation": "3fold_selection_plus_5fold_confirmation",
        "folds": N_SPLITS,
        **confirmacao,
        "selection_accuracy": float(grid.best_score_),
        "parameters": melhor_config,
        "notes": (
            f"Seleção exploratória com {N_SPLITS_EXPLORATORIA}-fold e "
            f"confirmação oficial com {N_SPLITS}-fold."
        ),
    }
    registrar_experimento(resultado)
    salvar_tabela(resultados, config["output"])

    print(f"\nTabela completa salva em: {config['output']}")
    return resultado, resultados


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa um dos experimentos de tuning do projeto."
    )
    parser.add_argument(
        "experimento",
        choices=EXPERIMENTS.keys(),
        help="Experimento de tuning a executar.",
    )
    args = parser.parse_args()
    executar_tuning(args.experimento)


if __name__ == "__main__":
    main()
