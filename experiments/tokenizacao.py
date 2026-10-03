import pandas as pd

from sklearn.model_selection import GridSearchCV

from src.config import (
    N_SPLITS,
    N_SPLITS_EXPLORATORIA,
    RANDOM_STATE,
)
from src.data import carregar_dados_treino
from src.models import criar_pipeline_tfidf_word_svc
from src.evaluation import criar_cv


X, y = carregar_dados_treino()


pipeline = criar_pipeline_tfidf_word_svc(
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    C=0.25,
)


param_grid = {
    "tfidf__lowercase": [
        True,
        False,
    ],

    "tfidf__token_pattern": [
        r"(?u)\b\w\w+\b",
        r"(?u)\b\w+\b",
    ],
}


grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=criar_cv(
        N_SPLITS_EXPLORATORIA
    ),
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    },
    refit="accuracy",
    n_jobs=-1,
    return_train_score=True,
    error_score="raise",
)


print("\n==============================")
print("EXPERIMENTO DE TOKENIZAÇÃO")
print("==============================")


grid.fit(X, y)


resultados = pd.DataFrame(
    grid.cv_results_
)


colunas = [
    "param_tfidf__lowercase",
    "param_tfidf__token_pattern",
    "mean_train_accuracy",
    "mean_test_accuracy",
    "std_test_accuracy",
    "mean_test_f1_macro",
]

resultados = resultados[
    colunas
].copy()


resultados.rename(
    columns={
        "param_tfidf__lowercase":
            "lowercase",

        "param_tfidf__token_pattern":
            "token_pattern",

        "mean_train_accuracy":
            "train_accuracy_3fold",

        "mean_test_accuracy":
            "accuracy_3fold",

        "std_test_accuracy":
            "accuracy_std_3fold",

        "mean_test_f1_macro":
            "f1_macro_3fold",
    },
    inplace=True,
)


resultados.sort_values(
    "accuracy_3fold",
    ascending=False,
    inplace=True,
)


print("\nResultados:")

print(
    resultados.to_string(
        index=False
    )
)


print("\nMelhores parâmetros:")

print(
    grid.best_params_
)


print(
    f"\nMelhor Accuracy 3-fold: "
    f"{grid.best_score_:.4f}"
)


resultados.to_csv(
    "results/tuning/tokenizacao.csv",
    index=False,
)


print(
    "\nTabela salva em:"
    " results/tuning/tokenizacao.csv"
)