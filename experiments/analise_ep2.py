"""Análise de características e de erros do modelo do EP2 (só com o treino).

Saídas em results/ep2/analise/:
- meta_coeficientes.csv: peso de cada meta-atributo (base x classe e
  atributos de texto) na meta-LR, com atributos padronizados;
- ngramas_por_classe.csv: n-gramas de maior peso por classe no TF-IDF;
- diversidade_bases.csv: acurácia OOF de cada base e taxa de discordância
  entre pares de bases (bases que erram diferente ajudam o stacking);
- diagnostico_stacking.csv / confusao_stacking.csv: recall, precisão,
  acurácia por tamanho e em duplicatas das predições OOF do stacking.

    python -m experiments.analise_ep2
"""

import json
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from experiments.stacking import META_MODELO_FILE
from src.config import CLASSES, EP2_ANALISE_DIR, EP2_RESULTS_DIR, RANDOM_STATE, STACKING_CONFIG_FILE
from src.data import carregar_dados_treino
from src.incremental import diagnosticar, matriz_confusao
from src.oof import carregar_base
from src.results import salvar_tabela

ATRIBUTOS_TEXTO = ["log_n_palavras", "tem_url", "menciona_anexo"]
TOP_N = 25


def coeficientes_meta(config) -> pd.DataFrame:
    modelo = joblib.load(META_MODELO_FILE)
    nomes = [f"logp_{base}_{c}" for base in sorted(config["bases"]) for c in CLASSES] + ATRIBUTOS_TEXTO
    coefs = modelo.named_steps["lr"].coef_
    return pd.DataFrame(coefs.T, index=nomes, columns=[f"coef_{c}" for c in modelo.classes_]).reset_index(names="atributo")


def ngramas_por_classe(X, y) -> pd.DataFrame:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    A = tfidf.fit_transform(X)
    lr = LogisticRegression(C=0.5, max_iter=3000, random_state=RANDOM_STATE).fit(A, y)
    vocab = np.array(tfidf.get_feature_names_out())
    linhas = []
    for i, classe in enumerate(lr.classes_):
        for rank, j in enumerate(np.argsort(lr.coef_[i])[::-1][:TOP_N], start=1):
            linhas.append({"classe": classe, "rank": rank, "ngrama": vocab[j], "coef": lr.coef_[i, j]})
    return pd.DataFrame(linhas)


def diversidade(config, y) -> pd.DataFrame:
    preds = {b: np.array(CLASSES)[carregar_base(b)["oof"].argmax(1)] for b in config["bases"]}
    linhas = [{"base_a": b, "base_b": "-", "acc_oof": (p == y).mean(), "discordancia": None} for b, p in preds.items()]
    for a, b in combinations(preds, 2):
        linhas.append({"base_a": a, "base_b": b, "acc_oof": None,
                       "discordancia": (preds[a] != preds[b]).mean(),
                       "so_a_acerta": ((preds[a] == y) & (preds[b] != y)).mean(),
                       "so_b_acerta": ((preds[a] != y) & (preds[b] == y)).mean()})
    return pd.DataFrame(linhas)


def main() -> None:
    EP2_ANALISE_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(STACKING_CONFIG_FILE.read_text(encoding="utf-8"))
    X, y = carregar_dados_treino()
    y_arr = y.to_numpy()

    if config["estrategia"] == "meta_lr":
        coefs = coeficientes_meta(config)
        salvar_tabela(coefs, EP2_ANALISE_DIR / "meta_coeficientes.csv")
        print("Meta-LR (atributos padronizados):\n" + coefs.round(3).to_string(index=False))

    ngramas = ngramas_por_classe(X, y)
    salvar_tabela(ngramas, EP2_ANALISE_DIR / "ngramas_por_classe.csv")
    for classe, grupo in ngramas.groupby("classe"):
        print(f"\nTop n-gramas {classe}: {', '.join(grupo['ngrama'].head(12))}")

    div = diversidade(config, y_arr)
    salvar_tabela(div, EP2_ANALISE_DIR / "diversidade_bases.csv")
    print("\n" + div.round(4).to_string(index=False))

    oof = pd.read_csv(EP2_RESULTS_DIR / "stacking_oof_pred.csv")
    diag = diagnosticar(X.to_numpy(dtype=object), oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
    salvar_tabela(diag, EP2_ANALISE_DIR / "diagnostico_stacking.csv")
    matriz_confusao(oof["y_true"], oof["y_pred"]).to_csv(EP2_ANALISE_DIR / "confusao_stacking.csv")
    print("\n" + diag.pivot_table(index="grupo", columns="valor", values="accuracy", sort=False).round(3).to_string())
    print("\n" + matriz_confusao(oof["y_true"], oof["y_pred"]).to_string())


if __name__ == "__main__":
    main()
