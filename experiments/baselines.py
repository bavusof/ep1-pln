import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.config import (
    BASELINE_RESULTS_FILE,
    BASELINE_TFIDF_GRID_FILE,
    CLASSES,
    METRICA_SELECAO,
    N_SPLITS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data import carregar_dados_treino
from src.evaluation import criar_grid_search, avaliar_modelo
from src.models import criar_baseline_majoritario, criar_pipeline_tfidf_word
from src.results import registrar_experimento, salvar_tabela, serializar_parametros


def mostrar_dados(X, y) -> None:
    print(f"Quantidade de exemplos: {len(X)}")
    print("\nDistribuição das classes:")
    print(y.value_counts())


def executar_baseline_majoritario(X, y) -> dict:
    """Baseline que sempre prevê a classe majoritária."""
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    modelo = criar_baseline_majoritario()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    classe_majoritaria = modelo.classes_[modelo.class_prior_.argmax()]

    print("\n==============================")
    print("BASELINE - CLASSE MAJORITÁRIA")
    print("==============================")
    print(f"Classe prevista: {classe_majoritaria}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"F1 Macro: {f1_macro:.4f}")

    resultado = {
        "experiment_id": "baseline_majority",
        "experiment_name": "Classe Majoritária",
        "category": "baseline",
        "representation": "Nenhuma",
        "model": "DummyClassifier",
        "evaluation": "holdout",
        "folds": None,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "parameters": {"strategy": "most_frequent"},
        "notes": f"Classe majoritária: {classe_majoritaria}",
    }
    registrar_experimento(resultado)
    return resultado


def executar_baseline_tfidf_holdout(X, y) -> dict:
    """Baseline oficial: TF-IDF padrão + Regressão Logística em holdout."""
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    modelo = criar_pipeline_tfidf_word()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    matriz = pd.DataFrame(
        confusion_matrix(y_val, y_pred, labels=CLASSES),
        index=[f"Real {classe}" for classe in CLASSES],
        columns=[f"Predito {classe}" for classe in CLASSES],
    )

    print("\n==============================")
    print("BASELINE OFICIAL - HOLDOUT")
    print("==============================")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"F1 Macro: {f1_macro:.4f}")
    print("\nMatriz de confusão:")
    print(matriz)
    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_val,
            y_pred,
            labels=CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    resultado = {
        "experiment_id": "baseline_tfidf_holdout",
        "experiment_name": "TF-IDF + Regressão Logística - Holdout",
        "category": "baseline",
        "representation": "TF-IDF palavra",
        "model": "Regressão Logística",
        "evaluation": "holdout",
        "folds": None,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "parameters": {
            "tfidf": "defaults",
            "C": 1.0,
            "max_iter": 1000,
        },
    }
    registrar_experimento(resultado)
    return resultado


def executar_baseline_tfidf_cv(X, y) -> tuple[dict, pd.DataFrame]:
    """Baseline oficial com a metodologia de 5-fold CV."""
    modelo = criar_pipeline_tfidf_word()
    _, resumo, resultados_folds = avaliar_modelo(modelo, X, y, n_splits=N_SPLITS)

    print("\n==============================")
    print("BASELINE - 5-FOLD CV")
    print("==============================")
    print(resultados_folds.to_string(index=False))
    print("\nResumo:")
    print(f"Accuracy: {resumo['accuracy']:.4f} ± {resumo['accuracy_std']:.4f}")
    print(f"F1 Macro: {resumo['f1_macro']:.4f} ± {resumo['f1_macro_std']:.4f}")
    print(f"Accuracy treino: {resumo['train_accuracy']:.4f}")
    print(f"F1 Macro treino: {resumo['train_f1_macro']:.4f}")

    resultado = {
        "experiment_id": "baseline_tfidf_cv",
        "experiment_name": "TF-IDF + Regressão Logística - 5-Fold CV",
        "category": "baseline",
        "representation": "TF-IDF palavra",
        "model": "Regressão Logística",
        "evaluation": "cross_validation",
        "folds": N_SPLITS,
        **resumo,
        "parameters": {
            "tfidf": "defaults",
            "C": 1.0,
            "max_iter": 1000,
        },
    }
    registrar_experimento(resultado)
    return resultado, resultados_folds


def executar_grid_tfidf_baseline(X, y) -> tuple[dict, pd.DataFrame]:
    """Grid Search aplciado no baseline"""
    modelo = criar_pipeline_tfidf_word()
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "logistic__C": [0.5, 1.0, 2.0],
    }

    grid = criar_grid_search(
        modelo,
        param_grid,
        n_splits=N_SPLITS,
        refit=METRICA_SELECAO,
        n_jobs=-1,
    )
    grid.fit(X, y)

    resultados = pd.DataFrame(grid.cv_results_)[
        [
            "param_tfidf__ngram_range",
            "param_logistic__C",
            "mean_train_accuracy",
            "std_train_accuracy",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_train_f1_macro",
            "mean_test_f1_macro",
        ]
    ].copy()

    resultados.rename(
        columns={
            "param_tfidf__ngram_range": "ngram_range",
            "param_logistic__C": "C",
            "mean_train_accuracy": "train_accuracy",
            "std_train_accuracy": "train_accuracy_std",
            "mean_test_accuracy": "accuracy",
            "std_test_accuracy": "accuracy_std",
            "mean_train_f1_macro": "train_f1_macro",
            "mean_test_f1_macro": "f1_macro",
        },
        inplace=True,
    )
    resultados["accuracy_gap"] = (
        resultados["train_accuracy"] - resultados["accuracy"]
    )
    resultados.sort_values("accuracy", ascending=False, inplace=True)
    resultados.reset_index(drop=True, inplace=True)

    print("\n==============================")
    print("GRID BASELINE - TF-IDF + REGRESSÃO LOGÍSTICA")
    print("==============================")
    print(resultados.to_string(index=False))

    melhor = resultados.iloc[0]
    resultado = {
        "experiment_id": "baseline_tfidf_grid",
        "experiment_name": "Grid Search - TF-IDF + Regressão Logística",
        "category": "baseline_grid",
        "representation": "TF-IDF palavra",
        "model": "Regressão Logística",
        "evaluation": "grid_search_cv",
        "folds": N_SPLITS,
        "accuracy": float(melhor["accuracy"]),
        "accuracy_std": float(melhor["accuracy_std"]),
        "f1_macro": float(melhor["f1_macro"]),
        "train_accuracy": float(melhor["train_accuracy"]),
        "train_f1_macro": float(melhor["train_f1_macro"]),
        "accuracy_gap": float(melhor["accuracy_gap"]),
        "selection_accuracy": float(grid.best_score_),
        "parameters": grid.best_params_,
    }
    registrar_experimento(resultado)
    salvar_tabela(resultados, BASELINE_TFIDF_GRID_FILE)
    return resultado, resultados


def main() -> None:
    X, y = carregar_dados_treino()
    mostrar_dados(X, y)

    resultados = []
    resultados.append(executar_baseline_majoritario(X, y))
    resultados.append(executar_baseline_tfidf_holdout(X, y))
    cv_resultado, _ = executar_baseline_tfidf_cv(X, y)
    resultados.append(cv_resultado)
    grid_resultado, _ = executar_grid_tfidf_baseline(X, y)
    resultados.append(grid_resultado)

    baseline_df = pd.DataFrame(resultados)
    baseline_df["parameters"] = baseline_df["parameters"].apply(serializar_parametros)
    salvar_tabela(baseline_df, BASELINE_RESULTS_FILE)

    print(f"\nResultados dos baselines salvos em: {BASELINE_RESULTS_FILE}")
    print(f"Grid do baseline salvo em: {BASELINE_TFIDF_GRID_FILE}")


if __name__ == "__main__":
    main()
