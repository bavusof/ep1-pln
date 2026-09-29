import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSES,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data import carregar_dados_treino
from registro_resultados import registrar_resultado


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

X, y = carregar_dados_treino()

print("Quantidade de exemplos:", len(X))
print("\nDistribuição das classes:")
print(y.value_counts())


# ============================================================
# DIVISÃO TREINO / VALIDAÇÃO
# ============================================================

# O stratify mantém aproximadamente a mesma proporção das
# classes nos conjuntos de treino e validação.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTamanho do treino:", len(X_train))
print("Tamanho da validação:", len(X_val))


# ============================================================
# PIPELINE BASELINE
# ============================================================

# O TF-IDF permanece dentro do Pipeline.
#
# Isso garante que o vocabulário e os pesos TF-IDF sejam
# aprendidos somente a partir dos dados de treinamento.
#
# Mesmo neste experimento simples, manter essa estrutura
# evita data leakage e deixa o código consistente com os
# experimentos posteriores.
pipeline_baseline = Pipeline([
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
# TREINAMENTO
# ============================================================

pipeline_baseline.fit(X_train, y_train)


# ============================================================
# PREDIÇÃO
# ============================================================

y_pred = pipeline_baseline.predict(X_val)


# ============================================================
# AVALIAÇÃO
# ============================================================

acuracia = accuracy_score(y_val, y_pred)

f1_macro = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0,
)


print("\n==============================")
print("RESULTADO DO BASELINE")
print("==============================")

print(
    f"Acurácia: {acuracia:.4f} "
    f"({acuracia * 100:.2f}%)"
)

print(f"F1 Macro: {f1_macro:.4f}")


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

matriz = confusion_matrix(
    y_val,
    y_pred,
    labels=CLASSES,
)

matriz_df = pd.DataFrame(
    matriz,
    index=[f"Real {classe}" for classe in CLASSES],
    columns=[f"Predito {classe}" for classe in CLASSES],
)

print("\nMatriz de confusão:")
print(matriz_df)


# ============================================================
# MÉTRICAS POR CLASSE
# ============================================================

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


# ============================================================
# REGISTRO DO RESULTADO
# ============================================================

resultado_baseline = {
    "experimento": "Holdout - Baseline",
    "representacao": "TF-IDF",
    "modelo": "Regressão Logística",
    "acuracia": acuracia,
    "f1_macro": f1_macro,
}

print("\nResultado:")
print(resultado_baseline)

registrar_resultado(resultado_baseline)