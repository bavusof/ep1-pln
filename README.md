# Projeto de PLN — Classificação de Clareza

Projeto desenvolvido para o EP1 da disciplina de **Introdução ao Processamento de Língua Natural**, com o objetivo de classificar a clareza de respostas provenientes de sistemas e-SIC.

A variável textual utilizada é `resp_text` e a variável alvo é `clarity`, com três classes:

- `c1`
- `c234`
- `c5`

O projeto foi estruturado para separar carregamento e processamento dos dados, implementação dos modelos, experimentos, avaliação e armazenamento dos resultados.

---

## Estrutura do projeto

```text
ep1-pln/
├── data/
│   ├── train.xlsx
│   └── test1.xlsx
│
├── notebooks/
│   ├── 01_exploracao.ipynb
│   └── 02_baselines.ipynb
│
├── experiments/
│   ├── __init__.py
│   ├── baselines.py
│   ├── current.py
│   ├── ensemble.py
│   ├── error_analysis.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── hierarchical_tfidf.py
│   ├── hybrid_tfidf_sentence_transformer.py
│   ├── hybrid_tfidf_word2vec.py
│   ├── linear_svc.py
│   ├── predict_test.py
│   ├── preprocessing.py
│   ├── sentence_transformer.py
│   ├── tfidf_weighted_word2vec.py
│   ├── tokenizacao.py
│   ├── token_weighted_bert.py
│   ├── tuning.py
│   └── word2vec.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── embeddings.py
│   ├── error_analysis.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   ├── results.py
│   ├── sentence_embeddings.py
│   ├── text_preprocessing.py
│   └── token_weighted_bert.py
│
├── results/
│   ├── baselines/
│   │   ├── resultados.csv
│   │   └── grid_tfidf.csv
│   │
│   ├── embeddings/
│   │   └── bertimbau_base_portuguese_sts.npy
│   │
│   ├── error_analysis/
│   │   ├── classification_report.csv
│   │   ├── erros.csv
│   │   ├── erros_confianca_alta.csv
│   │   ├── erros_confianca_baixa.csv
│   │   ├── matriz_confusao.csv
│   │   ├── predicoes_holdout.csv
│   │   └── resumo_confusoes.csv
│   │
│   ├── tuning/
│   │   ├── feature_engineering.csv
│   │   ├── feature_selection.csv
│   │   ├── hierarchical_tfidf.csv
│   │   ├── hybrid_tfidf_sentence_transformer.csv
│   │   ├── hybrid_tfidf_sentence_transformer_folds_5fold.csv
│   │   ├── hybrid_tfidf_word2vec.csv
│   │   ├── linear_svc_folds_5fold.csv
│   │   ├── linear_svc_grid.csv
│   │   ├── preprocessing.csv
│   │   ├── sentence_transformer.csv
│   │   ├── tfidf_char.csv
│   │   ├── tfidf_weighted_word2vec.csv
│   │   ├── tfidf_word.csv
│   │   ├── tfidf_word_char.csv
│   │   ├── tokenizacao.csv
│   │   ├── token_weighted_bert.csv
│   │   ├── token_weighted_bert_folds_5fold.csv
│   │   └── word2vec.csv
│   │
│   └── summary/
│       ├── experiments.csv
│       ├── current_model.json
│       └── final_model.json
│
├── requirements.txt
└── README.md
```

O arquivo `test1_rotulado.xlsx` é gerado na raiz do projeto pelo script de predição final.

---

## Requisitos

O código atual deve ser executado com **Python 3.10 ou superior**.

Alguns módulos utilizam recursos de tipagem que não são compatíveis com Python 3.8.

Para instalar as dependências:

```bash
python -m pip install -r requirements.txt
```

As principais bibliotecas utilizadas são:

- pandas;
- scikit-learn;
- openpyxl;
- gensim;
- sentence-transformers.

Na primeira execução de modelos baseados em Sentence Transformer pode ser necessário acesso à internet para baixar o modelo pré-treinado.

---

## Organização do código

### `src/`

Contém o código reutilizável do projeto.

`config.py` centraliza caminhos, classes, sementes, configurações de validação e arquivos de saída.

`data.py` realiza a leitura do `train.xlsx` e os tratamentos básicos. Valores ausentes em `resp_text` são substituídos por strings vazias e os textos são convertidos para `str`.

`evaluation.py` implementa a validação cruzada estratificada, avaliação dos modelos e criação dos objetos de `GridSearchCV`.

`models.py` contém os principais pipelines utilizados nos experimentos, incluindo TF-IDF, Regressão Logística, LinearSVC, Word2Vec, modelos híbridos, seleção de atributos e classificação hierárquica.

`embeddings.py` contém as implementações utilizadas para gerar representações documentais com Word2Vec e TF-IDF-weighted Word2Vec.

`features.py` implementa características estruturais do texto, como número de palavras, frases, URLs, números, riqueza lexical e proporção de letras maiúsculas.

`text_preprocessing.py` contém as funções de normalização textual utilizadas nos experimentos de pré-processamento.

`sentence_embeddings.py` é responsável pela geração e pelo cache dos embeddings do Sentence Transformer.

`token_weighted_bert.py` implementa a representação contextual em nível de token com pooling ponderado por TF-IDF.

`error_analysis.py` contém as funções utilizadas para análise das previsões, matriz de confusão e principais erros.

`results.py` centraliza o salvamento das tabelas, registro dos experimentos e arquivos de resumo.

---

## Experimentos

### Baselines

Os baselines podem ser executados com:

```bash
python -m experiments.baselines
```

São avaliados:

1. classificador de classe majoritária;
2. TF-IDF + Regressão Logística em holdout;
3. TF-IDF + Regressão Logística em validação cruzada;
4. Grid Search do baseline TF-IDF + Regressão Logística.

Os resultados são armazenados em:

```text
results/baselines/
```

---

## Tuning de TF-IDF

### TF-IDF de palavras

```bash
python -m experiments.tuning tfidf_word
```

### TF-IDF de caracteres

```bash
python -m experiments.tuning tfidf_char
```

### TF-IDF de palavras + caracteres

```bash
python -m experiments.tuning tfidf_word_char
```

A seleção das configurações é realizada inicialmente com validação cruzada estratificada de 3 folds e o melhor candidato é posteriormente confirmado com 5 folds.

---

## Outros experimentos

O projeto também contém experimentos específicos para outras representações e estratégias.

### Pré-processamento

```bash
python -m experiments.preprocessing
```

Compara diferentes estratégias de normalização, tratamento de URLs, números e stopwords.

### Tokenização

```bash
python -m experiments.tokenizacao
```

Avalia configurações de tokenização utilizadas pelo TF-IDF.

### Seleção de atributos

```bash
python -m experiments.feature_selection
```

Avalia a seleção de atributos TF-IDF utilizando qui-quadrado.

### Características estruturais

```bash
python -m experiments.feature_engineering
```

Combina TF-IDF com características estruturais extraídas dos textos.

### Word2Vec

```bash
python -m experiments.word2vec
```

Representa cada documento pela média dos embeddings Word2Vec.

### TF-IDF-weighted Word2Vec

```bash
python -m experiments.tfidf_weighted_word2vec
```

Utiliza os valores de TF-IDF para ponderar os embeddings Word2Vec na construção da representação documental.

### Híbrido TF-IDF + Word2Vec

```bash
python -m experiments.hybrid_tfidf_word2vec
```

Combina representação esparsa TF-IDF com embeddings Word2Vec.

### Sentence Transformer

```bash
python -m experiments.sentence_transformer
```

Utiliza embeddings contextuais pré-treinados do modelo:

```text
alfaneo/bertimbau-base-portuguese-sts
```

O encoder é utilizado como extrator fixo de características.

### Híbrido TF-IDF + Sentence Transformer

```bash
python -m experiments.hybrid_tfidf_sentence_transformer
```

Combina a representação TF-IDF com os embeddings contextuais produzidos pelo Sentence Transformer.

### Token-level TF-IDF-weighted BERT

```bash
python -m experiments.token_weighted_bert
```

Constrói uma representação documental a partir dos embeddings contextuais dos tokens, ponderados pelos respectivos valores de TF-IDF.

### Classificação hierárquica

```bash
python -m experiments.hierarchical_tfidf
```

O problema é dividido em duas etapas:

1. `c234` contra não-`c234`;
2. `c1` contra `c5`.

### LinearSVC

```bash
python -m experiments.linear_svc
```

Avalia a substituição da Regressão Logística por um classificador `LinearSVC` utilizando representação TF-IDF.

---

## Metodologia de avaliação

A configuração experimental padrão utiliza:

```text
random_state = 42
```

A validação é feita com `StratifiedKFold`, preservando aproximadamente a proporção das três classes entre os folds.

Para os experimentos com seleção de hiperparâmetros, o procedimento utilizado foi:

1. exploração inicial com 3 folds;
2. escolha da melhor configuração pela acurácia;
3. confirmação da configuração selecionada com 5 folds.

As principais métricas registradas são:

- Accuracy;
- Macro-F1;
- Accuracy de treinamento;
- Macro-F1 de treinamento;
- desvio padrão entre folds;
- gap entre accuracy de treinamento e validação.

O desempenho de treinamento é registrado principalmente para permitir a identificação de possíveis casos de sobreajuste.

---

## Resumo dos resultados

O classificador de classe majoritária apresentou aproximadamente:

```text
Accuracy: 0.3431
Macro-F1: 0.1703
```

O baseline TF-IDF + Regressão Logística em validação cruzada apresentou aproximadamente:

```text
Accuracy: 0.4572
Macro-F1: 0.4557
```

O Grid Search aplicado ao baseline alcançou aproximadamente:

```text
Accuracy: 0.4613
Macro-F1: 0.4602
```

A melhor configuração de TF-IDF encontrada durante o tuning ficou próxima de:

```text
Accuracy: 0.4619
Macro-F1: 0.4598
```

Grande parte dos experimentos realizados permaneceu na faixa de aproximadamente 0.45 a 0.46 de acurácia.

Representações como Word2Vec isolado apresentaram desempenho inferior, enquanto os modelos contextuais e híbridos ficaram mais próximos das melhores configurações de TF-IDF.

O melhor resultado médio entre os experimentos utilizados na seleção final foi obtido pelo ensemble de TF-IDF com Sentence Transformer.

---

# Modelo final da entrega

O modelo selecionado para gerar os rótulos do conjunto de teste é o:

**Ensemble TF-IDF + Sentence Transformer**

O ensemble combina as probabilidades produzidas por dois classificadores.

### Modelo A — TF-IDF

Representação:

```text
TF-IDF de palavras
```

Parâmetros utilizados:

```text
ngram_range = (1, 2)
min_df = 1
max_df = 1.0
sublinear_tf = True
C = 1.0
```

O classificador utilizado é uma Regressão Logística.

### Modelo B — TF-IDF + Sentence Transformer

O segundo classificador concatena:

```text
TF-IDF
+
embedding contextual
```

Os embeddings são produzidos pelo modelo:

```text
alfaneo/bertimbau-base-portuguese-sts
```

O Sentence Transformer é utilizado como extrator pré-treinado e congelado de características.

Os embeddings são multiplicados por:

```text
0.1
```

antes da concatenação com a representação TF-IDF.

A Regressão Logística do modelo híbrido utiliza:

```text
C = 0.5
```

### Combinação das probabilidades

A predição final é calculada pela média ponderada:

```text
P(final) =
    0.2 * P(TF-IDF)
    +
    0.8 * P(híbrido)
```

O rótulo final corresponde à classe de maior probabilidade entre:

```text
c1
c234
c5
```

---

## Desempenho do modelo final

A seleção do peso do ensemble foi realizada com validação cruzada estratificada de 3 folds.

A configuração selecionada foi posteriormente confirmada com 5 folds.

O resultado final foi aproximadamente:

```text
Accuracy: 0.4625
Macro-F1: 0.4611
```

A diferença em relação às melhores configurações baseadas somente em TF-IDF é pequena. Portanto, o resultado deve ser interpretado como a melhor média observada entre os experimentos avaliados, e não como uma melhoria substancial sobre o baseline.

---

## Gerando os rótulos do conjunto de teste

O arquivo de teste deve estar localizado em:

```text
data/test1.xlsx
```

A partir da raiz do projeto, execute:

```bash
python -m experiments.predict_test
```

O script:

1. carrega todo o conjunto de treinamento;
2. carrega o conjunto de teste;
3. gera ou reutiliza os embeddings do Sentence Transformer;
4. ajusta o TF-IDF utilizando todo o conjunto de treinamento;
5. treina o modelo TF-IDF;
6. treina o modelo híbrido;
7. combina as probabilidades utilizando os pesos definidos pelo ensemble;
8. gera os rótulos `c1`, `c234` ou `c5`;
9. valida a saída;
10. salva a planilha final.

O arquivo resultante é criado na raiz do projeto:

```text
test1_rotulado.xlsx
```

A saída contém somente:

```text
resp_text
clarity
```

A ordem dos textos é preservada.

---

## Cache dos embeddings

Os embeddings do conjunto de treinamento são armazenados em:

```text
results/embeddings/bertimbau_base_portuguese_sts.npy
```

O script de predição também cria um cache separado para o conjunto de teste.

Isso evita recalcular os embeddings em execuções posteriores.

---

## Análise de erros

A análise de erros do checkpoint configurado em `experiments/current.py` pode ser executada com:

```bash
python -m experiments.error_analysis
```

Ela utiliza uma divisão holdout estratificada de 20%, com:

```text
random_state = 42
```

Essa divisão é usada somente para inspeção dos erros e não substitui a validação cruzada utilizada para comparar os modelos.

Os resultados são armazenados em:

```text
results/error_analysis/
```

Arquivos produzidos:

```text
predicoes_holdout.csv
matriz_confusao.csv
classification_report.csv
resumo_confusoes.csv
erros.csv
erros_confianca_alta.csv
erros_confianca_baixa.csv
```

A análise inclui classe real, classe predita, probabilidades, confiança da previsão e características simples relacionadas ao tamanho dos textos.

É importante observar que essa análise utiliza a configuração presente em `experiments/current.py`, que corresponde a um checkpoint de desenvolvimento e não ao ensemble utilizado na entrega final.

---

## Resultados consolidados

O arquivo:

```text
results/summary/experiments.csv
```

contém o histórico consolidado dos principais experimentos realizados.

As tabelas detalhadas ficam em:

```text
results/baselines/
results/tuning/
```

### `current_model.json`

O arquivo:

```text
results/summary/current_model.json
```

representa o modelo configurado em `experiments/current.py`.

Ele foi mantido como registro de um checkpoint anterior do desenvolvimento.

### `final_model.json`

O arquivo:

```text
results/summary/final_model.json
```

documenta a configuração efetivamente utilizada para a entrega final.

Para evitar perder o histórico experimental, `current_model.json` não é sobrescrito pelo modelo final.

---

## Fluxo de reprodução da entrega

A partir da raiz do projeto:

```bash
python -m pip install -r requirements.txt
```

Em seguida, verifique se os arquivos estão presentes:

```text
data/train.xlsx
data/test1.xlsx
```

Por fim:

```bash
python -m experiments.predict_test
```

A saída esperada é:

```text
test1_rotulado.xlsx
```

---

## Nomes antigos → estrutura atual

| Antigo | Atual |
|---|---|
| `baselineClasseMajoritaria.py` | `experiments/baselines.py` |
| `baselineEP_RLTFIDF.py` | `experiments/baselines.py` |
| `baselineRLTFIDF_CrossValidation.py` | `experiments/baselines.py` |
| `rlTfidfFirstGridCv.py` | `experiments/baselines.py` |
| `rlTfidfSecondGridCv.py` | `experiments/tuning.py tfidf_word` |
| `rlTfidfCaracteresNgramGridCv.py` | `experiments/tuning.py tfidf_char` |
| `rlTidfPalavraCaractereNgramGridCv.py` | `experiments/tuning.py tfidf_word_char` |
| `experimentos_direcionados.py` | experimentos específicos em `experiments/` |
| `registro_resultados.py` | `src/results.py` |

---

## Observações

Os arquivos de tuning preservam o histórico dos experimentos realizados durante o desenvolvimento.

`experiments/current.py` continua disponível como mecanismo para avaliar uma configuração simples através da infraestrutura genérica definida em `src/models.py`.

O ensemble final possui uma implementação própria porque combina as probabilidades de dois modelos diferentes e, portanto, não é representado pelo mecanismo genérico de `CURRENT_MODEL`.

Para reproduzir especificamente a solução utilizada na entrega, utilize:

```bash
python -m experiments.predict_test
```