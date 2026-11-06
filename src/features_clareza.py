"""Características de clareza feitas à mão (não usam rótulos).

Grupos:
- legibilidade: Flesch adaptado ao português (Martins et al., 1996),
  palavras por frase e fração de palavras longas;
- jargão: termos jurídico-administrativos e siglas;
- conteúdo: recusa, encaminhamento, anexo em texto curto, prazo, estrutura.
"""

import re

import numpy as np
import pandas as pd

from src.text_preprocessing import normalizar_texto

PALAVRA = re.compile(r"[A-Za-zÀ-ÿ]+")
FRASE = re.compile(r"[.!?;]+")
VOGAIS = re.compile(r"[aeiouáéíóúâêôãõàü]+", re.IGNORECASE)
SIGLA = re.compile(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,}\b")

PADROES = {
    "jargao": r"\b(lei|leis|decreto|art\.?|artigo|inciso|par[aá]grafo|§|portaria|resolu[cç][aã]o|instru[cç][aã]o normativa|al[ií]nea|nup|sei)\b",
    "recusa": r"n[aã]o (foi poss[ií]vel|disp[oõ]e|possui|dispomos|consta|compete|cabe)|indeferid|sigil|informa[cç][aã]o inexistente|n[aã]o localizad",
    "encaminhamento": r"encaminh|procure|acesse|s[ií]tio|\blink\b|dispon[ií]vel em|consult[ae] o",
    "anexo": r"\banex",
    "prazo": r"\bprazo\b|dias [uú]teis|\brecurso\b",
    "lista": r"(^|\s)(\d{1,2}[.)]|[a-h]\))\s",
    "agradecimento": r"agradec|atenciosamente|cordialmente",
}

COLUNAS = [
    "flesch", "palavras_por_frase", "frac_palavras_longas", "log_palavras",
    "frac_jargao", "frac_siglas",
    "tem_recusa", "tem_encaminhamento", "anexo_texto_curto", "tem_prazo",
    "n_itens_lista", "tem_agradecimento", "frac_digitos",
]


def _silabas(palavra: str) -> int:
    return max(1, len(VOGAIS.findall(palavra)))


def _linha(texto: str) -> list[float]:
    texto = normalizar_texto(texto)
    minusculo = texto.lower()
    palavras = PALAVRA.findall(texto)
    n_palavras = max(1, len(palavras))
    n_frases = max(1, len([f for f in FRASE.split(texto) if f.strip()]))
    silabas = [_silabas(p) for p in palavras] or [1]
    palavras_por_frase = n_palavras / n_frases
    silabas_por_palavra = float(np.mean(silabas))
    flesch = 248.835 - 1.015 * palavras_por_frase - 84.6 * silabas_por_palavra
    conta = lambda chave: len(re.findall(PADROES[chave], minusculo))
    return [
        flesch,
        palavras_por_frase,
        float(np.mean([s >= 4 for s in silabas])),
        float(np.log1p(len(palavras))),
        conta("jargao") / n_palavras,
        len(SIGLA.findall(texto)) / n_palavras,
        float(conta("recusa") > 0),
        float(conta("encaminhamento") > 0),
        float(conta("anexo") > 0 and len(palavras) <= 40),
        float(conta("prazo") > 0),
        float(conta("lista")),
        float(conta("agradecimento") > 0),
        sum(c.isdigit() for c in texto) / max(1, len(texto)),
    ]


def atributos_clareza(textos) -> pd.DataFrame:
    textos = pd.Series(textos).fillna("").astype(str)
    return pd.DataFrame([_linha(t) for t in textos], columns=COLUNAS, index=textos.index)
