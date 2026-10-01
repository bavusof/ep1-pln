# Projeto de PLN — Classificação de Clareza

Estrutura reorganizada para separar código reutilizável, baselines, tuning e resultados.

## Estrutura

```text
projeto/
├── data/
│   └── train.xlsx
├── notebooks/
├── results/
│   ├── baselines/
│   │   ├── resultados.csv
│   │   └── grid_tfidf.csv
│   ├── tuning/
│   │   ├── tfidf_word.csv
│   │   ├── tfidf_char.csv
│   │   └── tfidf_word_char.csv
│   └── summary/
│       ├── experiments.csv
│       └── current_model.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── models.py
│   └── results.py
├── experiments/
│   ├── __init__.py
│   ├── baselines.py
│   ├── tuning.py
│   └── current.py
├── requirements.txt
└── README.md
```

## Responsabilidade dos módulos

### `src/`

Contém a lógica reutilizável:

- `config.py`: caminhos, classes, sementes, validação e arquivos de resultado.
- `data.py`: leitura do `train.xlsx` e tratamentos básicos.
- `evaluation.py`: validação cruzada e Grid Search.
- `models.py`: construção dos pipelines e do baseline majoritário.
- `results.py`: registro consolidado, CSVs e modelo atual.

### `experiments/`

Contém somente os experimentos que são executados.

- `baselines.py`: todos os baselines em um único lugar.
- `tuning.py`: todos os grids de exploração em um único executor.
- `current.py`: configuração que está sendo tratada como modelo atual.

## Como executar

A partir da raiz do projeto:

```bash
python -m experiments.baselines
```

Para tuning de TF-IDF por palavras:

```bash
python -m experiments.tuning tfidf_word
```

Para tuning de caracteres:

```bash
python -m experiments.tuning tfidf_char
```

Para tuning de palavra + caractere:

```bash
python -m experiments.tuning tfidf_word_char
```

Para avaliar o modelo atualmente configurado:

```bash
python -m experiments.current
```

## Fluxo recomendado para evolução do modelo

1. Rode o baseline para manter uma referência.
2. Execute um dos experimentos de tuning.
3. Veja a tabela gerada em `results/tuning/`.
4. Escolha a configuração que será testada como modelo atual.
5. Copie os parâmetros escolhidos para `CURRENT_MODEL` em `experiments/current.py`.
6. Rode `python -m experiments.current`.

A configuração do `current.py` não é alterada automaticamente pelo tuning. Isso evita que uma exploração experimental sobrescreva sem querer o modelo que o grupo está tratando como atual.

## Metodologia

Os grids de tuning mantêm a lógica dos códigos originais:

- seleção exploratória com 3-fold estratificado;
- confirmação do melhor candidato com 5-fold estratificado;
- `accuracy` como métrica de seleção do Grid Search;
- `F1 macro` também é registrado para comparação;
- `random_state=42` para reprodutibilidade.

O baseline oficial em holdout também foi preservado, assim como o baseline de classe majoritária.

## Resultados

`results/summary/experiments.csv` é a tabela consolidada de experimentos.

`results/summary/current_model.json` guarda a configuração e as métricas do modelo atualmente controlado por `experiments/current.py`.

As tabelas detalhadas dos grids ficam em `results/baselines/` e `results/tuning/`.

## Nomes antigos → nomes novos

| Antigo | Novo |
|---|---|
| `baselineClasseMajoritaria.py` | `experiments/baselines.py` |
| `baselineEP_RLTFIDF.py` | `experiments/baselines.py` |
| `baselineRLTFIDF_CrossValidation.py` | `experiments/baselines.py` |
| `rlTfidfFirstGridCv.py` | `experiments/baselines.py` |
| `rlTfidfSecondGridCv.py` | `experiments/tuning.py tfidf_word` |
| `rlTfidfCaracteresNgramGridCv.py` | `experiments/tuning.py tfidf_char` |
| `rlTidfPalavraCaractereNgramGridCv.py` | `experiments/tuning.py tfidf_word_char` |
| `experimentos_direcionados.py` | `experiments/tuning.py` + `experiments/current.py` |
| `registro_resultados.py` | `src/results.py` |

## Observação

O projeto não exige um novo arquivo Python para cada nova tentativa de ajuste. Para explorar novas configurações, altere o `param_grid` de `experiments/tuning.py`. Para acompanhar uma configuração específica como modelo atual, altere apenas `CURRENT_MODEL` em `experiments/current.py`.
