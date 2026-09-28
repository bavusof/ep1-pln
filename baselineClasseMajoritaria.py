import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


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

print("\nDistribuição das classes:")
print(df[COLUNA_CLASSE].value_counts())


# ============================================================
# 3. SEPARAÇÃO ENTRE ENTRADA (X) E CLASSE (y)
# ============================================================

X = df[COLUNA_TEXTO]

y = df[COLUNA_CLASSE]


# ============================================================
# 4. DIVISÃO TREINO / VALIDAÇÃO
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# 5. DUMMY CLASSIFIER
# ============================================================

modelo = DummyClassifier(
    strategy="most_frequent"
)

modelo.fit(X_train, y_train)


# ============================================================
# 6. PREDIÇÃO
# ============================================================

y_pred = modelo.predict(X_val)


# ============================================================
# 7. AVALIAÇÃO
# ============================================================

f1 = f1_score(
    y_val,
    y_pred,
    average="macro"
)

print("\n==============================")

print("BASELINE - CLASSE MAJORITÁRIA")

print("==============================")

print(
    "Classe escolhida pelo modelo:",
    modelo.classes_[modelo.class_prior_.argmax()]
)

print(f"F1 macro: {f1:.4f}")

print(f"F1 macro (%): {f1 * 100:.2f}%")


# ============================================================
# 8. REGISTRO DO RESULTADO
# ============================================================

resultado_baseline = {
    "modelo": "Dummy - Classe Majoritária",
    "f1_macro": f1
}

print("\nResultado salvo:")

print(resultado_baseline)