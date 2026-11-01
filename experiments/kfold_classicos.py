"""Bases clássicas do stacking do EP2, nos mesmos 5 folds dos transformers.

- `tfidf`: melhor modelo clássico da escada incremental (degrau 6:
  TF-IDF de palavras com bigramas + características estruturais + LR).
- `ep1_ensemble`: o modelo entregue no EP1 (0,2 x TF-IDF + 0,8 x híbrido
  TF-IDF + Sentence-BERT congelado), refeito fold a fold.

    python -m experiments.kfold_classicos [tfidf] [ep1_ensemble]
"""

import copy
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from experiments.ensemble import C_HYBRID, C_TFIDF, EMBEDDING_WEIGHT, TFIDF_PARAMS
from experiments.predict_test import FINAL_HYBRID_BLEND_WEIGHT, _hash_textos
from src.config import (
    CLASSES,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    SENTENCE_TRANSFORMER_BATCH_SIZE,
    SENTENCE_TRANSFORMER_EMBEDDINGS_FILE,
    SENTENCE_TRANSFORMER_MODEL_NAME,
    SENTENCE_TRANSFORMER_NORMALIZE,
)
from src.data import carregar_dados_treino
from src.incremental import CONFIG_INICIAL, construir_pipeline
from src.oof import acuracia_por_fold, carregar_textos_teste, consolidar, folds_ep2, salvar_fold
from src.results import registrar_experimento
from src.sentence_embeddings import carregar_ou_gerar_embeddings
from src.ruido_rotulos import mascara_manter
from src.tratamento_dados import tratar_treino


# Degrau 6 da escada incremental (melhor clássico: 0,4619 no CV agrupado).
CONFIG_TFIDF = copy.deepcopy(CONFIG_INICIAL)
CONFIG_TFIDF["word"]["ngram_range"] = (1, 2)
CONFIG_TFIDF["structural_weight"] = 1.0


def _ordenar(modelo, probs):
    return probs[:, [list(modelo.classes_).index(c) for c in CLASSES]]


def _fold_tfidf(X, y, treino, validacao, testes, limpo=False, fold=None, frac_cl=None):
    X_tr, y_tr = X[treino], y[treino]
    if limpo:
        X_tr, y_tr, _ = tratar_treino(X_tr, y_tr)
    if frac_cl is not None:
        # Confident learning: remove a fração de menor qualidade, estimada só com o treino do fold.
        manter, rel = mascara_manter(X_tr, y_tr, frac_cl, fold=fold)
        print(f"fold {fold}: cleanlab removeu {rel['removidas']} linhas {rel['removidas_por_classe']}", flush=True)
        X_tr, y_tr = X_tr[manter], y_tr[manter]
    modelo = construir_pipeline(CONFIG_TFIDF).fit(X_tr, y_tr)
    prever = lambda textos: _ordenar(modelo, modelo.predict_proba(textos))
    return validacao, prever(X[validacao]), prever(testes["test1"]), prever(testes["test2"])


def embeddings(textos, nome):
    """Embeddings do Sentence-BERT do EP1 (cache por hash dos textos)."""
    arquivo = (
        SENTENCE_TRANSFORMER_EMBEDDINGS_FILE
        if nome == "train"
        else RESULTS_DIR / "embeddings" / f"bertimbau_base_portuguese_sts_test_{_hash_textos(textos)}.npy"
    )
    return carregar_ou_gerar_embeddings(
        textos,
        model_name=SENTENCE_TRANSFORMER_MODEL_NAME,
        output_file=arquivo,
        batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
        normalize_embeddings=SENTENCE_TRANSFORMER_NORMALIZE,
    )


def _fold_ep1(X, y, emb, treino, validacao, testes, emb_testes):
    """Mesma receita de experiments/predict_test.py, ajustada só no treino do fold."""
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    A_treino = tfidf.fit_transform(X[treino])
    lr_tfidf = LogisticRegression(C=C_TFIDF, max_iter=3000, random_state=RANDOM_STATE).fit(A_treino, y[treino])
    hibrido = lambda A, E: hstack([A, csr_matrix(E * EMBEDDING_WEIGHT)], format="csr")
    lr_hib = LogisticRegression(C=C_HYBRID, max_iter=3000, random_state=RANDOM_STATE).fit(
        hibrido(A_treino, emb[treino]), y[treino]
    )

    def prever(textos, E):
        A = tfidf.transform(textos)
        p_tfidf = _ordenar(lr_tfidf, lr_tfidf.predict_proba(A))
        p_hib = _ordenar(lr_hib, lr_hib.predict_proba(hibrido(A, E)))
        return (1 - FINAL_HYBRID_BLEND_WEIGHT) * p_tfidf + FINAL_HYBRID_BLEND_WEIGHT * p_hib

    return (
        validacao,
        prever(X[validacao], emb[validacao]),
        prever(testes["test1"], emb_testes["test1"]),
        prever(testes["test2"], emb_testes["test2"]),
    )


def executar(tag, X, y, splits, testes) -> None:
    X_arr, y_arr = X.to_numpy(dtype=object), y.to_numpy()
    testes_arr = {k: v.to_numpy(dtype=object) for k, v in testes.items()}
    if tag in ("tfidf", "tfidf_limpo") or tag.startswith("tfidf_cl"):
        limpo = tag == "tfidf_limpo"
        frac_cl = int(tag.removeprefix("tfidf_cl")) / 100 if tag.startswith("tfidf_cl") else None
        tarefas = [delayed(_fold_tfidf)(X_arr, y_arr, tr, va, testes_arr, limpo, fold, frac_cl)
                   for fold, (tr, va) in enumerate(splits)]
    else:
        emb = embeddings(X, "train")
        emb_testes = {k: embeddings(v, k) for k, v in testes.items()}
        tarefas = [delayed(_fold_ep1)(X_arr, y_arr, emb, tr, va, testes_arr, emb_testes) for tr, va in splits]

    # n_jobs pequeno para não disputar CPU com o fine-tuning em paralelo.
    for fold, (validacao, p_val, p_t1, p_t2) in enumerate(Parallel(n_jobs=2)(tarefas)):
        salvar_fold(tag, fold, val_idx=validacao, probs_val=p_val, probs_test1=p_t1, probs_test2=p_t2)

    base = consolidar(tag, len(y_arr), len(splits))
    acc = acuracia_por_fold(base["oof"], base["fold_ids"], y_arr)
    pred = np.array(CLASSES)[base["oof"].argmax(1)]
    print(f"{tag}: acurácia OOF = {acc.mean():.4f} ± {acc.std():.4f} | por fold: {np.round(acc, 4)}")
    registrar_experimento(
        {
            "experiment_id": f"ep2_oof_{tag}",
            "experiment_name": f"EP2 base {tag} (5-fold OOF)",
            "category": "ep2_base",
            "representation": "TF-IDF + estruturais" if tag.startswith("tfidf") else "TF-IDF + Sentence-BERT (EP1)",
            "model": "Regressão Logística" if tag.startswith("tfidf") else "Ensemble do EP1",
            "evaluation": "5-fold StratifiedGroupKFold por texto (OOF)",
            "folds": N_SPLITS,
            "accuracy": float(acc.mean()),
            "accuracy_std": float(acc.std()),
            "f1_macro": float(f1_score(y_arr, pred, average="macro")),
            "notes": "Previsões de teste = média dos 5 modelos dos folds."
                     + (" Treino de cada fold tratado (src/tratamento_dados.py)." if tag.endswith("_limpo") else "")
                     + (f" Cleanlab: {tag.removeprefix('tfidf_cl')}% das linhas de menor qualidade removidas do treino de cada fold." if "_cl" in tag else ""),
        }
    )


def main() -> None:
    tags = sys.argv[1:] or ["tfidf", "ep1_ensemble"]
    X, y = carregar_dados_treino()
    splits = folds_ep2(X, y)
    testes = carregar_textos_teste()
    for tag in tags:
        executar(tag, X, y, splits, testes)


if __name__ == "__main__":
    main()
