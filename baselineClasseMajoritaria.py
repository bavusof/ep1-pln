from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import (
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

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# BASELINE DE CLASSE MAJORITÁRIA
# ============================================================

# Este modelo não utiliza o conteúdo textual.
#
# Ele simplesmente aprende qual é a classe mais frequente
# no conjunto de treinamento e sempre prevê essa classe.
#
# O objetivo é estabelecer uma referência simples para
# verificar se os modelos de PLN realmente aprendem algo
# além da distribuição das classes.
modelo = DummyClassifier(
    strategy="most_frequent",
)

modelo.fit(X_train, y_train)


# ============================================================
# PREDIÇÃO
# ============================================================

y_pred = modelo.predict(X_val)


# ============================================================
# AVALIAÇÃO
# ============================================================

acuracia = accuracy_score(
    y_val,
    y_pred,
)

f1_macro = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0,
)

classe_majoritaria = (
    modelo.classes_[modelo.class_prior_.argmax()]
)


print("\n==============================")
print("BASELINE - CLASSE MAJORITÁRIA")
print("==============================")

print(
    "Classe escolhida pelo modelo:",
    classe_majoritaria,
)

print(
    f"Acurácia: {acuracia:.4f} "
    f"({acuracia * 100:.2f}%)"
)

print(f"F1 Macro: {f1_macro:.4f}")


# ============================================================
# REGISTRO DO RESULTADO
# ============================================================

resultado_baseline = {
    "experimento": "Classe Majoritária",
    "representacao": "Nenhuma",
    "modelo": "DummyClassifier",
    "acuracia": acuracia,
    "f1_macro": f1_macro,
}

print("\nResultado:")
print(resultado_baseline)

registrar_resultado(resultado_baseline)