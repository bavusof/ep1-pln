from pathlib import Path


# ============================================================
# CONFIGURAÇÕES GERAIS DO PROJETO
# ============================================================

# Diretório raiz do projeto.
# Mantemos os caminhos centralizados para evitar que cada
# experimento tenha uma configuração diferente.
ROOT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DADOS
# ============================================================

ARQUIVO_TREINO = ROOT_DIR / "data" / "train.xlsx"

COLUNA_TEXTO = "resp_text"
COLUNA_CLASSE = "clarity"

# Ordem oficial das classes utilizada nas avaliações.
# Mantemos a ordem definida pelo problema: c1, c234, c5.
CLASSES = ["c1", "c234", "c5"]


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# VALIDAÇÃO
# ============================================================

TEST_SIZE = 0.20
N_SPLITS = 5


# ============================================================
# MÉTRICAS
# ============================================================

METRICAS = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
}