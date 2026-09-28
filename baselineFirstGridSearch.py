import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from registro_resultados import registrar_resultado


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

ARQUIVO_TREINO = "train.xlsx"

COLUNA_TEXTO = "resp_text"
COLUNA_CLASSE = "clarity"

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

df = pd.read_excel(ARQUIVO_TREINO)

df[COLUNA_TEXTO] = df[COLUNA_TEXTO].fillna("").astype(str)

print("Dimensões:", df.shape)

print("\nColunas:")
print(df.columns.tolist())

print("\nDistribuição das classes:")
print(df[COLUNA_CLASSE].value_counts())

X = df[COLUNA_TEXTO]
y = df[COLUNA_CLASSE]


# ============================================================
# 3. VALIDAÇÃO CRUZADA
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 4. PIPELINE
# ============================================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("logistic", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])


# ============================================================
# 5. PARÂMETROS A SEREM TESTADOS
# ============================================================

param_grid = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2)
    ],
    "logistic__C": [
        0.5,
        1.0,
        2.0
    ]
}


# ============================================================
# 6. GRID SEARCH
# ============================================================

# A acurácia é usada para escolher o melhor modelo,
# pois é a métrica principal da avaliação oficial.
# F1 macro é calculado simultaneamente como complemento.

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    },
    refit="accuracy",
    n_jobs=2,
    pre_dispatch=2,
    return_train_score=True,
    error_score="raise"
)


print("\n==============================")
print("INICIANDO GRID SEARCH")
print("==============================")

grid_search.fit(X, y)


# ============================================================
# 7. MELHOR RESULTADO
# ============================================================

print("\n==============================")
print("MELHOR RESULTADO")
print("==============================")

print("Melhores parâmetros:")
print(grid_search.best_params_)

melhor_acuracia = float(grid_search.best_score_)

print(f"\nMelhor acurácia média: {melhor_acuracia:.4f}")


# ============================================================
# 8. RESULTADOS DE TODAS AS CONFIGURAÇÕES
# ============================================================

resultados = pd.DataFrame(grid_search.cv_results_)

resultados = resultados[
    [
        "param_tfidf__ngram_range",
        "param_logistic__C",
        "mean_train_accuracy",
        "std_train_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_train_f1_macro",
        "mean_test_f1_macro"
    ]
].copy()

resultados = resultados.rename(columns={
    "param_tfidf__ngram_range": "ngram_range",
    "param_logistic__C": "C",
    "mean_train_accuracy": "accuracy_treino_media",
    "std_train_accuracy": "desvio_accuracy_treino",
    "mean_test_accuracy": "accuracy_validacao_media",
    "std_test_accuracy": "desvio_accuracy_validacao",
    "mean_train_f1_macro": "f1_treino_medio",
    "mean_test_f1_macro": "f1_validacao_medio"
})

resultados = resultados.sort_values(
    by="accuracy_validacao_media",
    ascending=False
).reset_index(drop=True)


print("\n==============================")
print("TODOS OS RESULTADOS")
print("==============================")

print(resultados.to_string(index=False))


# ============================================================
# 9. INDICADOR SIMPLES DE OVERFITTING
# ============================================================

resultados["gap_accuracy"] = (
    resultados["accuracy_treino_media"]
    - resultados["accuracy_validacao_media"]
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
            "gap_accuracy"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. REGISTRO DO MELHOR RESULTADO
# ============================================================

melhor_linha = resultados.iloc[0]

resultado_grid = {
    "experimento": "Grid Search - TF-IDF + Regressão Logística",
    "representacao": "TF-IDF",
    "modelo": "Regressão Logística",

    # Conversão explícita de tipos NumPy para tipos Python
    "acuracia_media": float(grid_search.best_score_),
    "f1_macro_medio": float(melhor_linha["f1_validacao_medio"]),
    "gap_accuracy": float(melhor_linha["gap_accuracy"]),

    # Converte valores dos parâmetros para tipos Python
    "melhores_parametros": {
        "tfidf__ngram_range": tuple(
            int(valor)
            for valor in grid_search.best_params_["tfidf__ngram_range"]
        ),
        "logistic__C": float(
            grid_search.best_params_["logistic__C"]
        )
    }
}

print("\nResultado:")
print(resultado_grid)

registrar_resultado(resultado_grid)


# ============================================================
# 11. CONVERSÃO DOS RESULTADOS PARA TIPOS PYTHON
# ============================================================

# Garante que valores numéricos da tabela sejam tipos Python
# antes do salvamento.

colunas_numericas = [
    "C",
    "accuracy_treino_media",
    "desvio_accuracy_treino",
    "accuracy_validacao_media",
    "desvio_accuracy_validacao",
    "f1_treino_medio",
    "f1_validacao_medio",
    "gap_accuracy"
]

for coluna in colunas_numericas:
    resultados[coluna] = resultados[coluna].astype(float)


# ============================================================
# 12. SALVAMENTO DA TABELA DE EXPERIMENTOS
# ============================================================

resultados.to_csv(
    "resultados_grid_baseline.csv",
    index=False
)

print("\nTabela salva em: resultados_grid_baseline.csv")
