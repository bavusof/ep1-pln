from pathlib import Path


# ============================================================
# DIRETÓRIOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
BASELINES_RESULTS_DIR = RESULTS_DIR / "baselines"
TUNING_RESULTS_DIR = RESULTS_DIR / "tuning"
SUMMARY_RESULTS_DIR = RESULTS_DIR / "summary"

for directory in (
    DATA_DIR,
    RESULTS_DIR,
    BASELINES_RESULTS_DIR,
    TUNING_RESULTS_DIR,
    SUMMARY_RESULTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# DADOS
# ============================================================

ARQUIVO_TREINO = DATA_DIR / "train.xlsx"
COLUNA_TEXTO = "resp_text"
COLUNA_CLASSE = "clarity"

# Ordem oficial das classes utilizada nas avaliações.
CLASSES = ["c1", "c234", "c5"]


# ============================================================
# REPRODUTIBILIDADE E VALIDAÇÃO
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_SPLITS_EXPLORATORIA = 3


# Critério usado pelo GridSearchCV para selecionar a configuração.
METRICA_SELECAO = "accuracy"

METRICAS = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
}


# ============================================================
# RESULTADOS
# ============================================================

BASELINE_RESULTS_FILE = BASELINES_RESULTS_DIR / "resultados.csv"
BASELINE_TFIDF_GRID_FILE = BASELINES_RESULTS_DIR / "grid_tfidf.csv"

TUNING_WORD_FILE = TUNING_RESULTS_DIR / "tfidf_word.csv"
TUNING_CHAR_FILE = TUNING_RESULTS_DIR / "tfidf_char.csv"
TUNING_WORD_CHAR_FILE = TUNING_RESULTS_DIR / "tfidf_word_char.csv"

EXPERIMENTS_SUMMARY_FILE = SUMMARY_RESULTS_DIR / "experiments.csv"
CURRENT_MODEL_FILE = SUMMARY_RESULTS_DIR / "current_model.json"
