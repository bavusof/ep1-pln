import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    TUNING_HIERARCHICAL_TFIDF_FILE,
)
from src.data import carregar_dados_treino
from src.evaluation import avaliar_modelo, criar_cv
from src.models import HierarchicalClarityClassifier
from src.results import registrar_experimento, salvar_tabela


BASE_PARAMS = {
    "word_ngram_range": (1, 2),
    "word_min_df": 5,
    "word_max_df": 1.0,
    "word_sublinear_tf": True,
}


X, y = carregar_dados_treino()

pipeline = HierarchicalClarityClassifier(
    **BASE_PARAMS,
    C_c234=0.5,
    C_c1_c5=0.5,
    c234_threshold=0.5,
)


param_grid = {
    "C_c234": [0.25, 0.5, 1.0],
    "C_c1_c5": [0.25, 0.5, 1.0],
    "c234_threshold": [0.40, 0.50, 0.60],
}


print("\n==============================")
print("CLASSIFICAÇÃO HIERÁRQUICA")
print("==============================")
print("Estratégia: c234 vs. não-c234 -> c1 vs. c5")
print("Configurações:", 3 * 3 * 3)
print("Folds de exploração:", N_SPLITS_EXPLORATORIA)


grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=criar_cv(N_SPLITS_EXPLORATORIA),
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    },
    refit="accuracy",
    n_jobs=-1,
    return_train_score=True,
    error_score="raise",
)

grid.fit(X, y)


resultados = pd.DataFrame(grid.cv_results_)

resultados = resultados[
    [
        "param_C_c234",
        "param_C_c1_c5",
        "param_c234_threshold",
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
        "param_C_c234": "C_c234",
        "param_C_c1_c5": "C_c1_c5",
        "param_c234_threshold": "c234_threshold",
        "mean_train_accuracy": "train_accuracy_3fold",
        "std_train_accuracy": "train_accuracy_std_3fold",
        "mean_test_accuracy": "accuracy_3fold",
        "std_test_accuracy": "accuracy_std_3fold",
        "mean_train_f1_macro": "train_f1_macro_3fold",
        "mean_test_f1_macro": "f1_macro_3fold",
    },
    inplace=True,
)

resultados["accuracy_gap_3fold"] = (
    resultados["train_accuracy_3fold"]
    - resultados["accuracy_3fold"]
)

resultados.sort_values(
    "accuracy_3fold",
    ascending=False,
    inplace=True,
)

resultados.reset_index(
    drop=True,
    inplace=True,
)

print("\n==============================")
print("RESULTADOS - 3-FOLD")
print("==============================")
print(resultados.to_string(index=False))


_, resumo, resultados_folds = avaliar_modelo(
    grid.best_estimator_,
    X,
    y,
    n_splits=N_SPLITS,
)

melhor_config = {
    **BASE_PARAMS,
    "C_c234": float(grid.best_params_["C_c234"]),
    "C_c1_c5": float(grid.best_params_["C_c1_c5"]),
    "c234_threshold": float(
        grid.best_params_["c234_threshold"]
    ),
}

print("\n==============================")
print("CONFIRMAÇÃO - 5-FOLD")
print("==============================")
print("Melhor configuração:")
print(melhor_config)
print("\nResultados por fold:")
print(resultados_folds.to_string(index=False))
print(
    f"\nAccuracy: {resumo['accuracy']:.4f} "
    f"± {resumo['accuracy_std']:.4f}"
)
print(
    f"F1 Macro: {resumo['f1_macro']:.4f} "
    f"± {resumo['f1_macro_std']:.4f}"
)
print(
    f"Accuracy treino: "
    f"{resumo['train_accuracy']:.4f}"
)
print(
    f"Gap: {resumo['accuracy_gap']:.4f}"
)


registrar_experimento(
    {
        "experiment_id": "tuning_hierarchical_tfidf",
        "experiment_name": "TF-IDF + Classificação Hierárquica",
        "category": "model_architecture",
        "representation": "TF-IDF palavra",
        "model": "Regressão Logística hierárquica",
        "evaluation": "3fold_selection_plus_5fold_confirmation",
        "folds": N_SPLITS,
        **resumo,
        "selection_accuracy": float(grid.best_score_),
        "parameters": melhor_config,
        "notes": (
            "Primeiro classificador separa c234 de não-c234; "
            "o segundo separa c1 de c5. "
            "C e limiar de c234 selecionados em 3 folds "
            "e confirmados em 5 folds."
        ),
    }
)

salvar_tabela(
    resultados,
    TUNING_HIERARCHICAL_TFIDF_FILE,
)

print(
    "\nTabela salva em:",
    TUNING_HIERARCHICAL_TFIDF_FILE,
)
