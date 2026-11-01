"""LLM instruído como avaliador zero-shot da clareza (atributos para o stacking).

Para cada resposta, o LLM recebe 3 perguntas e lemos a probabilidade do
PRÓXIMO token (um único forward por pergunta, sem geração):
- nota: "De 1 a 5, quão clara é..." -> P("1"), ..., P("5") e nota esperada;
- entrega: "fornece diretamente a informação...?" -> P("Sim");
- dificil: "usa linguagem técnica ou jurídica difícil...?" -> P("Sim").

Os atributos não usam os rótulos, então são calculados uma vez para treino
e testes. Cache incremental por hash do texto (retomável).

    python -m experiments.llm_avaliador               # treino + testes
    python -m experiments.llm_avaliador --amostra 300 # checagem de sinal/velocidade
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.atributos_extras import LLM_COLUNAS as COLUNAS
from src.atributos_extras import atributos_llm, carregar_cache_llm as carregar_cache
from src.atributos_extras import chave_texto as chave, salvar_cache_llm as salvar_cache
from src.config import CLASSES
from src.data import carregar_dados_treino
from src.oof import carregar_textos_teste
from src.text_preprocessing import limpar_para_bert

MODELO = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS_TEXTO = 384
HEAD_TOKENS = 128

SISTEMA = (
    "Você é um avaliador de respostas do Serviço de Informação ao Cidadão (e-SIC). "
    "Avalie apenas a resposta dada ao cidadão."
)
PERGUNTAS = {
    "nota": "De 1 a 5, quão clara é esta resposta para um cidadão comum? "
            "(1 = nada clara, 5 = muito clara). Responda só com o número.",
    "entrega": "A resposta fornece diretamente a informação solicitada, em vez de apenas negar, "
               "encaminhar ou remeter a um anexo? Responda apenas Sim ou Não.",
    "dificil": "A resposta usa linguagem técnica ou jurídica difícil para um cidadão comum? "
               "Responda apenas Sim ou Não.",
}




def truncar(tokenizer, texto: str) -> str:
    """Mantém início e fim de respostas longas (mesma ideia de tokenizar_head_tail)."""
    ids = tokenizer(texto, add_special_tokens=False)["input_ids"]
    if len(ids) <= MAX_TOKENS_TEXTO:
        return texto
    ids = ids[:HEAD_TOKENS] + ids[-(MAX_TOKENS_TEXTO - HEAD_TOKENS):]
    return tokenizer.decode(ids)


def montar_prompt(tokenizer, texto: str, pergunta: str) -> str:
    mensagens = [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": f"Resposta ao cidadão:\n\"\"\"\n{texto}\n\"\"\"\n\n{pergunta}"},
    ]
    return tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)


def ids_resposta(tokenizer) -> dict[str, list[int]]:
    primeiro = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"][0]
    return {"nota": [primeiro(str(i)) for i in range(1, 6)], "sim_nao": [primeiro("Sim"), primeiro("Não")]}


@torch.no_grad()
def avaliar_lote(model, tokenizer, textos, ids, device) -> np.ndarray:
    saida = np.zeros((len(textos), len(COLUNAS)), dtype=np.float32)
    for nome, pergunta in PERGUNTAS.items():
        prompts = [montar_prompt(tokenizer, t, pergunta) for t in textos]
        lote = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        logits = model(**lote).logits[:, -1, :].float()  # padding à esquerda: último token = início da resposta
        if nome == "nota":
            p = torch.softmax(logits[:, ids["nota"]], dim=-1).cpu().numpy()
            saida[:, 0:5] = p
            saida[:, 5] = p @ np.arange(1, 6)
        else:
            p_sim = torch.softmax(logits[:, ids["sim_nao"]], dim=-1)[:, 0].cpu().numpy()
            saida[:, 6 if nome == "entrega" else 7] = p_sim
    return saida








def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostra", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    X, y = carregar_dados_treino()
    if args.amostra:
        amostra = pd.DataFrame({"x": X, "y": y}).sample(args.amostra, random_state=0)
        textos, rotulos = amostra["x"].tolist(), amostra["y"].to_numpy()
    else:
        testes = carregar_textos_teste()
        textos = X.tolist() + testes["test1"].tolist() + testes["test2"].tolist()
        rotulos = None

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODELO, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16).to(device).eval()
    ids = ids_resposta(tokenizer)
    print(f"{MODELO} | device={device} | tokens de resposta: {ids}", flush=True)

    limpos = [limpar_para_bert(t) for t in textos]
    cache = carregar_cache()
    pendentes = sorted({chave(t): t for t in limpos if chave(t) not in cache}.items(), key=lambda kv: len(kv[1]))
    print(f"{len(limpos)} textos | {len(pendentes)} únicos a avaliar | {len(cache)} já no cache", flush=True)

    inicio, desde_salvo = time.time(), 0
    for i in range(0, len(pendentes), args.batch_size):
        lote = pendentes[i : i + args.batch_size]
        valores = avaliar_lote(model, tokenizer, [truncar(tokenizer, t) for _, t in lote], ids, device)
        for (k, _), v in zip(lote, valores):
            cache[k] = v
        desde_salvo += len(lote)
        if desde_salvo >= 500 or i + args.batch_size >= len(pendentes):
            salvar_cache(cache)
            desde_salvo = 0
            feitos = i + len(lote)
            ritmo = (time.time() - inicio) / feitos
            print(f"  {feitos}/{len(pendentes)} ({ritmo:.2f} s/texto, faltam ~{ritmo * (len(pendentes) - feitos) / 60:.0f} min)", flush=True)

    if rotulos is not None:
        A = pd.DataFrame(atributos_llm(textos), columns=COLUNAS)
        A["classe"] = rotulos
        print("\nMédia por classe (amostra):")
        print(A.groupby("classe")[["llm_nota", "llm_entrega", "llm_dificil"]].mean().round(3).reindex(CLASSES).to_string())
        print("Soma P(1..5):", A[COLUNAS[:5]].sum(axis=1).round(4).describe()[["min", "max"]].to_dict())


if __name__ == "__main__":
    main()
