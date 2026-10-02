import re


# ============================================================
# CORREÇÃO DE CARACTERES FREQUENTEMENTE ENCONTRADOS NO CORPUS
# ============================================================

CP1252_REPLACEMENTS = str.maketrans(
    {
        "\x91": "‘",
        "\x92": "’",
        "\x93": "“",
        "\x94": "”",
        "\x95": "•",
        "\x96": "–",
        "\x97": "—",
        "\x85": "…",
    }
)


# ============================================================
# STOPWORDS EM PORTUGUÊS
# ============================================================

# Mantemos negações como "não", "nunca" e "nem",
# pois elas podem carregar informação semântica relevante.
STOPWORDS_PT = {
    "a", "à", "ao", "aos", "as",
    "às", "é", "e",
    "da", "das", "de", "do", "dos",
    "em", "na", "nas", "no", "nos",
    "por", "pela", "pelas", "pelo", "pelos",
    "para", "pra",
    "com", "sem",
    "sobre", "sob",
    "entre",
    "um", "uma", "uns", "umas",

    "o", "os",
    "que", "qual", "quais",
    "quem",
    "onde",
    "como",

    "se", "sua", "suas",
    "seu", "seus",
    "este", "esta", "estes", "estas",
    "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aqueles", "aquelas",
    "isto", "isso", "aquilo",

    "me", "te", "lhe", "nos", "vos",
    "meu", "minha", "meus", "minhas",
    "teu", "tua", "teus", "tuas",
    "nosso", "nossa", "nossos", "nossas",
    "vosso", "vossa", "vossos", "vossas",

    "já", "ainda",
    "também", "muito", "muita", "muitos", "muitas",
    "mais", "menos",
    "mesmo", "mesma", "mesmos", "mesmas",
    "outro", "outra", "outros", "outras",
    "cada", "todo", "toda", "todos", "todas",
    "outrem",

    "até", "desde",
    "após", "antes",
    "durante",
    "perante",
    "contra",
    "assim",
    "então",

    "ter", "tem", "têm",
    "há",
    "ser", "são",
    "foi", "foram",
    "era", "eram",
    "sendo",
    "está", "estão",
    "estava", "estavam",

    "pode", "podem",
    "poderá", "poderão",
    "deve", "devem",
    "deverá", "deverão",

    "quando",
    "porque",
    "pois",
    "portanto",
}


# ============================================================
# NORMALIZAÇÃO CONSERVADORA
# ============================================================

def normalizar_texto(texto: str) -> str:
    """
    Normalização conservadora.

    Não remove palavras, números ou pontuação.
    Apenas corrige caracteres problemáticos e espaços.
    """

    texto = str(texto)

    texto = texto.translate(
        CP1252_REPLACEMENTS
    )

    # Espaço não separável.
    texto = texto.replace(
        "\xa0",
        " ",
    )

    # Remove caracteres de controle, mantendo texto legível.
    texto = "".join(
        caractere
        for caractere in texto
        if ord(caractere) >= 32
        or caractere in "\n\t"
    )

    # Normaliza espaços.
    texto = re.sub(
        r"\s+",
        " ",
        texto,
    )

    return texto.strip()


# ============================================================
# NORMALIZAÇÃO DE URLs
# ============================================================

def normalizar_urls(texto: str) -> str:
    """
    Substitui URLs por um token único.

    O objetivo é evitar que o modelo aprenda padrões muito
    específicos de cada endereço eletrônico.
    """

    texto = normalizar_texto(texto)

    texto = re.sub(
        r"(https?://|www\.)\S+",
        " URLTOKEN ",
        texto,
        flags=re.IGNORECASE,
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto,
    )

    return texto.strip()


# ============================================================
# NORMALIZAÇÃO DE URLs + NÚMEROS
# ============================================================

def normalizar_urls_numeros(texto: str) -> str:
    """
    Além de URLs, substitui sequências numéricas por NUMTOKEN.

    Isso reduz a dependência de protocolos, números de processos,
    telefones, datas e identificadores específicos.
    """

    texto = normalizar_urls(texto)

    texto = re.sub(
        r"\b\d+(?:[./-]\d+)*\b",
        " NUMTOKEN ",
        texto,
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto,
    )

    return texto.strip()