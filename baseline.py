import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix, classification_report


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

df[COLUNA_TEXTO] = df[COLUNA_TEXTO].fillna("").astype(str)

print("Dimensões:", df.shape)

print("\nColunas:")
print(df.columns.tolist())

print("\nDistribuição das classes:")
print(df[COLUNA_CLASSE].value_counts())

X = df[COLUNA_TEXTO]

y = df[COLUNA_CLASSE]


# ============================================================
# 3. DIVISÃO ENTRE TREINO E VALIDAÇÃO
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
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
# 7. AVALIAÇÃO - F1
# ============================================================

f1 = f1_score(
    y_val,
    y_pred,
    average="macro"
)

print("\n==============================")

print("RESULTADO DO BASELINE")

print("==============================")

print(f"F1 macro: {f1:.4f}")


# ============================================================
# 8. MATRIZ DE CONFUSÃO
# ============================================================

matriz = confusion_matrix(
    y_val,
    y_pred,
    labels=["c1", "c234", "c5"]
)

print("\nMatriz de confusão:")

print(pd.DataFrame(
    matriz,
    index=["Real c1", "Real c234", "Real c5"],
    columns=["Predito c1", "Predito c234", "Predito c5"]
))


# ============================================================
# 9. MÉTRICAS POR CLASSE
# ============================================================

print("\nRelatório de classificação:")

print(classification_report(
    y_val,
    y_pred,
    labels=["c1", "c234", "c5"],
    digits=4
))


# ============================================================
# 10. REGISTRO DO RESULTADO
# ============================================================

resultado_baseline = {
    "modelo": "TF-IDF + Regressão Logística",
    "f1_macro": f1
}

print("\nResultado salvo:")

print(resultado_baseline)


# DataFrame da matriz, caso seja necessário posteriormente
matriz_df = pd.DataFrame(
    matriz,
    index=["Real c1", "Real c234", "Real c5"],
    columns=["Predito c1", "Predito c234", "Predito c5"]
)