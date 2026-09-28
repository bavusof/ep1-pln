import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

ARQUIVO_TREINO = "train.xlsx"

COLUNA_TEXTO = "resp_text"

COLUNA_CLASSE = "clarity"

RANDOM_STATE = 42


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

df = pd.read_excel(ARQUIVO_TREINO)

# Garante que todos os textos sejam tratados como strings
df[COLUNA_TEXTO] = df[COLUNA_TEXTO].fillna("").astype(str)

print("Dimensões:", df.shape)

print("\nColunas:")
print(df.columns.tolist())

print("\nDistribuição das classes:")
print(df[COLUNA_CLASSE].value_counts())


# ============================================================
# 3. SEPARAÇÃO ENTRE ENTRADA (X) E CLASSE (y)
# ============================================================

X = df[COLUNA_TEXTO]

y = df[COLUNA_CLASSE]


# ============================================================
# 4. VALIDAÇÃO CRUZADA
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 5. PIPELINE
# ============================================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("logistic", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])


# ============================================================
# 6. PARÂMETROS A SEREM TESTADOS
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
# 7. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=2,
    pre_dispatch=2,
    return_train_score=True,
    error_score="raise"
)


print("\n==============================")
print("INICIANDO GRID SEARCH - F1 MACRO")
print("==============================")

grid_search.fit(X, y)


# ============================================================
# 8. MELHOR RESULTADO
# ============================================================

print("\n==============================")
print("MELHOR RESULTADO")
print("==============================")

print("Melhores parâmetros:")
print(grid_search.best_params_)

print(f"\nMelhor F1 macro médio: {grid_search.best_score_:.4f}")


# ============================================================
# 9. RESULTADOS DE TODAS AS CONFIGURAÇÕES
# ============================================================

resultados = pd.DataFrame(grid_search.cv_results_)

resultados = resultados[
    [
        "param_tfidf__ngram_range",
        "param_logistic__C",
        "mean_test_score",
        "std_test_score",
        "mean_train_score"
    ]
].copy()

resultados = resultados.sort_values(
    by="mean_test_score",
    ascending=False
)

print("\n==============================")
print("TODOS OS RESULTADOS")
print("==============================")

print(resultados.to_string(index=False))


# ============================================================
# 10. REGISTRO DO RESULTADO
# ============================================================

resultado_grid = {
    "modelo": "TF-IDF + Regressão Logística",
    "metrica": "F1 macro",
    "f1_macro_medio": grid_search.best_score_,
    "melhores_parametros": grid_search.best_params_
}

print("\nResultado salvo:")
print(resultado_grid)