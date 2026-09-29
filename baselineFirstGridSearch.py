import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import (
    N_SPLITS,
    RANDOM_STATE,
)
from src.data import carregar_dados_treino
from src.evaluation import criar_cv
from registro_resultados import registrar_resultado


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

X, y = carregar_dados_treino()

print("Quantidade de exemplos:", len(X))

print("\nDistribuição das classes:")
print(y.value_counts())


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(),
    ),
    (
        "logistic",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    ),
])


# ============================================================
# PARÂMETROS A SEREM TESTADOS
# ============================================================

# Esta é uma primeira exploração do espaço de hiperparâmetros.
#
# Posteriormente poderemos expandir o grid com parâmetros
# como min_df, max_df e sublinear_tf.
#
# É importante fazer isso gradualmente para manter o custo
# computacional controlado.

param_grid = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2),
    ],

    "logistic__C": [
        0.5,
        1.0,
        2.0,
    ],
}


# ============================================================
# GRID SEARCH
# ============================================================

# A Accuracy é utilizada como métrica principal para escolher
# a configuração final do Grid Search.
#
# O F1 Macro é calculado simultaneamente como métrica
# complementar para acompanhar o comportamento das classes.
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=criar_cv(),

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
print("INICIANDO GRID SEARCH")
print("==============================")


grid_search.fit(X, y)


# ============================================================
# MELHOR RESULTADO
# ============================================================

print("\n==============================")
print("MELHOR RESULTADO")
print("==============================")

print("Melhores parâmetros:")
print(grid_search.best_params_)

melhor_acuracia = float(
    grid_search.best_score_
)

print(
    f"\nMelhor acurácia média: "
    f"{melhor_acuracia:.4f}"
)


# ============================================================
# RESULTADOS DE TODAS AS CONFIGURAÇÕES
# ============================================================

resultados = pd.DataFrame(
    grid_search.cv_results_
)

resultados = resultados[
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


resultados = resultados.rename(
    columns={
        "param_tfidf__ngram_range":
            "ngram_range",

        "param_logistic__C":
            "C",

        "mean_train_accuracy":
            "accuracy_treino_media",

        "std_train_accuracy":
            "desvio_accuracy_treino",

        "mean_test_accuracy":
            "accuracy_validacao_media",

        "std_test_accuracy":
            "desvio_accuracy_validacao",

        "mean_train_f1_macro":
            "f1_treino_medio",

        "mean_test_f1_macro":
            "f1_validacao_medio",
    }
)


# Ordenamos pela métrica utilizada para selecionar
# o melhor modelo.
resultados = (
    resultados
    .sort_values(
        by="accuracy_validacao_media",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# GAP TREINO - VALIDAÇÃO
# ============================================================

# O gap não é uma medida isolada de qualidade.
# Ele serve como indicador auxiliar para observar se uma
# configuração apresenta uma diferença grande entre treino
# e validação.
resultados["gap_accuracy"] = (
    resultados["accuracy_treino_media"]
    - resultados["accuracy_validacao_media"]
)


# ============================================================
# EXIBIÇÃO DOS RESULTADOS
# ============================================================

print("\n==============================")
print("TODOS OS RESULTADOS")
print("==============================")

print(
    resultados.to_string(index=False)
)


print("\n==============================")
print("GAP TREINO - VALIDAÇÃO")
print("==============================")

print(
    resultados[
        [
            "ngram_range",
            "C",
            "accuracy_treino_media",
            "accuracy_validacao_media",
            "gap_accuracy",
        ]
    ].to_string(index=False)
)


# ============================================================
# REGISTRO DO MELHOR RESULTADO
# ============================================================

melhor_linha = resultados.iloc[0]

melhores_parametros = {
    "tfidf__ngram_range": tuple(
        int(valor)
        for valor in grid_search.best_params_[
            "tfidf__ngram_range"
        ]
    ),

    "logistic__C": float(
        grid_search.best_params_[
            "logistic__C"
        ]
    ),
}


resultado_grid = {
    "experimento":
        "Grid Search - TF-IDF + Regressão Logística",

    "representacao":
        "TF-IDF",

    "modelo":
        "Regressão Logística",

    "folds":
        N_SPLITS,

    "acuracia_media":
        float(grid_search.best_score_),

    "f1_macro_medio":
        float(
            melhor_linha["f1_validacao_medio"]
        ),

    "gap_accuracy":
        float(
            melhor_linha["gap_accuracy"]
        ),

    "melhores_parametros":
        melhores_parametros,
}


print("\nResultado:")
print(resultado_grid)

registrar_resultado(resultado_grid)


# ============================================================
# CONVERSÃO DOS RESULTADOS
# ============================================================

# Alguns valores retornados pelo scikit-learn podem ser tipos
# NumPy. A conversão facilita o salvamento e o processamento
# posterior da tabela.
colunas_numericas = [
    "C",
    "accuracy_treino_media",
    "desvio_accuracy_treino",
    "accuracy_validacao_media",
    "desvio_accuracy_validacao",
    "f1_treino_medio",
    "f1_validacao_medio",
    "gap_accuracy",
]

for coluna in colunas_numericas:
    resultados[coluna] = resultados[coluna].astype(float)


# ============================================================
# SALVAMENTO
# ============================================================

resultados.to_csv(
    "results/resultados_grid_baseline.csv",
    index=False,
)

print(
    "\nTabela salva em: "
    "resultados_grid_baseline.csv"
)