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
TUNING_FEATURE_SELECTION_FILE = TUNING_RESULTS_DIR / "feature_selection.csv"
TUNING_PREPROCESSING_FILE = TUNING_RESULTS_DIR / "preprocessing.csv"
TUNING_TFIDF_WEIGHTED_WORD2VEC_FILE = TUNING_RESULTS_DIR / "tfidf_weighted_word2vec.csv"
TUNING_HYBRID_TFIDF_WORD2VEC_FILE = TUNING_RESULTS_DIR / "hybrid_tfidf_word2vec.csv"
TUNING_HIERARCHICAL_TFIDF_FILE = TUNING_RESULTS_DIR / "hierarchical_tfidf.csv"
TUNING_SENTENCE_TRANSFORMER_FILE = TUNING_RESULTS_DIR / "sentence_transformer.csv"
TUNING_HYBRID_TFIDF_SENTENCE_TRANSFORMER_FILE = TUNING_RESULTS_DIR / "hybrid_tfidf_sentence_transformer.csv"
TUNING_HYBRID_TFIDF_SENTENCE_TRANSFORMER_FOLDS_FILE = TUNING_RESULTS_DIR / "hybrid_tfidf_sentence_transformer_folds_5fold.csv"

# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

SENTENCE_TRANSFORMER_MODEL_NAME = "alfaneo/bertimbau-base-portuguese-sts"
SENTENCE_TRANSFORMER_BATCH_SIZE = 16
SENTENCE_TRANSFORMER_NORMALIZE = True
SENTENCE_TRANSFORMER_EMBEDDINGS_FILE = RESULTS_DIR / "embeddings" / "bertimbau_base_portuguese_sts.npy"

# ============================================================
# TOKEN-LEVEL TF-IDF-WEIGHTED BERT
# ============================================================

TOKEN_WEIGHTED_BERT_BATCH_SIZE = 16
TOKEN_WEIGHTED_BERT_LENGTHS = [256, 512]
TUNING_TOKEN_WEIGHTED_BERT_FILE = TUNING_RESULTS_DIR / "token_weighted_bert.csv"
TUNING_TOKEN_WEIGHTED_BERT_FOLDS_FILE = TUNING_RESULTS_DIR / "token_weighted_bert_folds_5fold.csv"

# ============================================================
# RESUMO E MODELO ATUAL
# ============================================================

EXPERIMENTS_SUMMARY_FILE = SUMMARY_RESULTS_DIR / "experiments.csv"
CURRENT_MODEL_FILE = SUMMARY_RESULTS_DIR / "current_model.json"

# ============================================================
# ERROR ANALYSIS
# ============================================================

ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"
