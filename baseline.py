import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from registro_resultados import registrar_resultado


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================
ARQUIVO_TREINO = "train.xlsx"
COLUNA_TEXTO = "resp_text"
COLUNA_CLASSE = "clarity"
CLASSES = ["c1", "c234", "c5"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


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
# 3. DIVISÃO TREINO / VALIDAÇÃO
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTamanho do treino:", len(X_train))
print("Tamanho da validação:", len(X_val))


# ============================================================
# 4. REPRESENTAÇÃO TF-IDF
# ============================================================
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print("\nDimensão da matriz TF-IDF:")
print("Treino:", X_train_tfidf.shape)
print("Validação:", X_val_tfidf.shape)


# ============================================================
# 5. REGRESSÃO LOGÍSTICA
# ============================================================
modelo = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)

modelo.fit(X_train_tfidf, y_train)


# ============================================================
# 6. PREDIÇÃO
# ============================================================
y_pred = modelo.predict(X_val_tfidf)


# ============================================================
# 7. AVALIAÇÃO
# ============================================================
acuracia = accuracy_score(y_val, y_pred)
f1_macro = f1_score(y_val, y_pred, average="macro")

print("\n==============================")
print("RESULTADO DO BASELINE")
print("==============================")
print(f"Acurácia:  {acuracia:.4f} ({acuracia * 100:.2f}%)")
print(f"F1 macro:  {f1_macro:.4f}")


# ============================================================
# 8. MATRIZ DE CONFUSÃO
# ============================================================
matriz = confusion_matrix(y_val, y_pred, labels=CLASSES)

matriz_df = pd.DataFrame(
    matriz,
    index=["Real c1", "Real c234", "Real c5"],
    columns=["Predito c1", "Predito c234", "Predito c5"]
)

print("\nMatriz de confusão:")
print(matriz_df)


# ============================================================
# 9. MÉTRICAS POR CLASSE
# ============================================================
print("\nRelatório de classificação:")
print(
    classification_report(
        y_val,
        y_pred,
        labels=CLASSES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 10. REGISTRO DO RESULTADO
# ============================================================
resultado_baseline = {
    "experimento": "Holdout - Baseline",
    "representacao": "TF-IDF",
    "modelo": "Regressão Logística",
    "acuracia": acuracia,
    "f1_macro": f1_macro
}

print("Resultado:")
print(resultado_baseline)
registrar_resultado(resultado_baseline)
