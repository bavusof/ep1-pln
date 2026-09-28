import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from registro_resultados import registrar_resultado


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================
ARQUIVO_TREINO = "train.xlsx"
COLUNA_TEXTO = "resp_text"
COLUNA_CLASSE = "clarity"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================
df = pd.read_excel(ARQUIVO_TREINO)
df[COLUNA_TEXTO] = df[COLUNA_TEXTO].fillna("").astype(str)

print("Dimensões:", df.shape)
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


# ============================================================
# 4. BASELINE DE CLASSE MAJORITÁRIA
# ============================================================
modelo = DummyClassifier(strategy="most_frequent")
modelo.fit(X_train, y_train)


# ============================================================
# 5. PREDIÇÃO
# ============================================================
y_pred = modelo.predict(X_val)


# ============================================================
# 6. AVALIAÇÃO
# ============================================================
acuracia = accuracy_score(y_val, y_pred)
f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)

classe_majoritaria = modelo.classes_[modelo.class_prior_.argmax()]

print("\n==============================")
print("BASELINE - CLASSE MAJORITÁRIA")
print("==============================")
print("Classe escolhida pelo modelo:", classe_majoritaria)
print(f"Acurácia:  {acuracia:.4f} ({acuracia * 100:.2f}%)")
print(f"F1 macro:  {f1_macro:.4f}")


# ============================================================
# 7. REGISTRO DO RESULTADO
# ============================================================
resultado_baseline = {
    "experimento": "Classe Majoritária",
    "representacao": "Nenhuma",
    "modelo": "DummyClassifier",
    "acuracia": acuracia,
    "f1_macro": f1_macro
}

print("Resultado:")
print(resultado_baseline)
registrar_resultado(resultado_baseline)
