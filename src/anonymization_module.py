# src/anonymization_module.py

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import Dict, List

# Modelo LENERBR
MODEL_NAME = "pierreguillou/ner-bert-large-cased-pt-lenerbr"

# Mapeamento de rótulos para tags genéricas (simplificado)
TAG_MAP: Dict[str, str] = {
    "PESSOA": "[ANON_PESSOA]",
    "ORGANIZACAO": "[ANON_ORG]",
    "LEGISLACAO": "[ANON_LEG]",
    "LOCAL": "[ANON_LOC]",
    "TEMPO": "[ANON_DATA]",
    "PROCESSO": "[ANON_NUM]",
    "CASE NUMBER": "[ANON_NUM]",
    "CPF": "[ANON_NUM]",
    "CNPJ": "[ANON_NUM]",
    "CREA": "[ANON_NUM]",
    "RG": "[ANON_NUM]",
    "ENDERECO": "[ANON_LOC]",
    # Outras classes permanecem com o rótulo da entidade
}


def load_ner_pipeline(model_name: str = MODEL_NAME):
    """Carrega um modelo NER pré-treinado a partir do Hugging Face."""
    print(f"  -> Carregando modelo NER LENER-Br: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    except Exception as e:
        print(f"❌ ERRO: Não foi possível carregar o pipeline NER. Instale 'transformers' e 'torch'. Erro: {e}")
        return None

# Variável global para armazenar o pipeline NER carregado
_NER_PIPE = None

def get_ner_pipeline():
    """Retorna o pipeline NER, carregando-o se ainda não estiver na memória."""
    global _NER_PIPE
    if _NER_PIPE is None:
        _NER_PIPE = load_ner_pipeline()
    return _NER_PIPE


def anonymize_text(text: str, ner_pipe) -> str:
    """
    Substitui entidades nomeadas por tags genéricas (o cerne do seu código).
    Divide o texto em blocos para contornar o limite de tokens do Transformer.
    """
    if not ner_pipe:
        return text

    anonymized_text = text
    
    # Parâmetros de chunking (ajustados para a lógica do código fornecido)
    block_size = 1000 
    
    # 1. Cria blocos de texto
    blocks = []
    start = 0
    while start < len(text):
        end = min(start + block_size, len(text))
        blocks.append((start, end, text[start:end]))
        start = end

    # 2. Processa os blocos de trás para frente para evitar problemas de índice
    for block_start, block_end, block_text in reversed(blocks):
        try:
            entities = ner_pipe(block_text)
        except Exception:
            # Lógica de fallback para blocos menores (do seu código)
            entities = []
            sub_block_size = 500
            sub_start = block_start
            while sub_start < block_end:
                sub_end = min(sub_start + sub_block_size, block_end)
                sub_text = text[sub_start:sub_end]
                try:
                    sub_entities = ner_pipe(sub_text)
                except Exception:
                    sub_entities = []
                for ent in sub_entities:
                    ent["start"] += sub_start - block_start
                    ent["end"] += sub_start - block_start
                entities.extend(sub_entities)
                sub_start = sub_end
                
        # 3. Substitui entidades no texto original
        for ent in reversed(entities):
            start_idx, end_idx = ent["start"], ent["end"]
            label = ent["entity_group"]
            tag = TAG_MAP.get(label, f"[ANON_{label}]") # Use o rótulo do LENERBR se não mapeado
            
            global_start = block_start + start_idx
            global_end = block_start + end_idx
            
            # Substituição
            anonymized_text = anonymized_text[:global_start] + tag + anonymized_text[global_end:]
            
    return anonymized_text