# src/llm_module.py (Atualização Focada no Prompt Detalhado)

import os
import json
from google import genai
from google.genai import types

# Mantém a função get_gemini_client()
def get_gemini_client():
    """Tenta inicializar o cliente Gemini usando a variável de ambiente."""
    try:
        # Tenta inicializar o cliente
        return genai.Client()
    except Exception as e:
        print(f"Erro ao inicializar o Gemini Client: {e}")
        return None


def gemini_process_pdf_file(pdf_path: str, client: genai.Client):
    """
    Faz o upload do PDF, usa o Gemini para extrair e estruturar informações,
    e deleta o arquivo temporário no servidor.
    """
    if not client:
        print("Não foi possível conectar ao Gemini. Pulando extração LLM.")
        return None

    uploaded_file = None
    try:
        # 1. UPLOAD do arquivo para a API do Gemini
        print(f"  -> Fazendo upload do arquivo: {pdf_path}")
        uploaded_file = client.files.upload(file=pdf_path)
        print(f"  -> Upload completo. URI: {uploaded_file.uri}")

        # 2. PROMPT DE EXTRAÇÃO E ANONIMIZAÇÃO CUSTOMIZADA
        prompt = f"""
        Você é um perito em engenharia elétrica e um especialista em PLN. 
        Analise o processo judicial contido no arquivo PDF anexado.
        
        Sua principal tarefa é extrair e estruturar as informações em formato JSON e, 
        ao mesmo tempo, realizar uma ANONIMIZAÇÃO E PREPARAÇÃO de texto seguindo as regras abaixo:
        
        --- REGRAS DE ANONIMIZAÇÃO E RESUMO ---
        
        1. ANONIMIZAÇÃO CUSTOMIZADA:
            a. SUBSTITUIR todos os nomes de pessoas, CPFs, RGs, e números de telefone por: [ANON_PESSOA]
            b. SUBSTITUIR CNPJ, Inscrição Estadual, e quaisquer números de documento por: [ANON_DOC]
            c. SUBSTITUIR nomes de empresas, escritórios ou organizações privadas por: [ANON_EMPRESA]
            d. SUBSTITUIR endereços, nomes de ruas, cidades e bairros por: [ANON_LOCAL]
            e. SUBSTITUIR números de registro de classe (CREA, OAB, CRM, etc.) por: [ANON_CLASSE]
            f. SUBSTITUIR números de processo ou matrículas por: [ANON_NUM_PROCESSO]
            

        2. FOCO DO RESUMO:
            O resumo principal deve ter pelo menos 25 linhas para a inicial e 25 linhas para a contestação ('sumario_fatos') e deve focar exclusivamente nos fatos descritos na PETIÇÃO INICIAL e na CONTESTAÇÃO. Somente pegue informações adicionais se forem de suma importância, como anexos por exemplo.

        3. RETENÇÃO DE TERMOS:
            No resumo e na extração de termos, MANTENHA todas as palavras técnicas da área de engenharia elétrica, eletrônica, de software, computação e telecomunicações(e.g., sobretensão, aterramento, disjuntor, neutro) e todas as palavras com alta carga emocional (e.g., sofrimento, descaso, negligência) para posterior análise de sentimentos e PLN, isso vale para todo o processo.
            Segue uma lista de ermos técnicos e emocionais a serem mantidos, de exemplo:
            termos técnicos = [
        'transformador', 'tensão', 'sobrecarga', 'curto', 'circuito', 'perícia',
        'laudo', 'equipamento', 'rede', 'energia', 'instalação', 'eletricidade',
        'queda', 'variação', 'proteção', 'concessionária', 'cabo', 'linha', 'pique','pico',
        'sistema', 'aparelho', 'disjuntor', 'resistência', 'potência', 'voltagem',
        'frequência', 'oxidação','presença de líquido','tensão','placa de vídeo','placa mãe','memória ram','hz','bytes','hd','ram','rom,','placa',
        'defeito','normas','normas técnicas','resolução','ABNT','ANEEL','corrente','voltagem','isolamento','curto-circuito',
        'fusível','transformador','capacitor','resistor','indutor','solda','conector', 'cabos', 'média tensão','alta tensão','baixa tensão', 'sobretensão','oscilação','aterramento','disjuntor','DR','DPS','NBR','protocolo de teste','ensaio','laudo técnico', 'prova pericial',
        'perito', 'perícia técnica', 'análise técnica', 'avaliação técnica', 'teste elétrico', 'medição elétrica', 'falha elétrica', 'curto circuito','subtensão', 'dispositivo de proteção', 'sistema elétrico', 'rede elétrica', 'equipamento elétrico'
    ]
    termos emocionais = [
        'danos', 'morais', 'sofrimento', 'angústia', 'transtorno', 'insegurança',
        'aflição', 'raiva', 'ansiedade', 'humilhação', 'dor', 'prejuízo', 'luto', 'danos sofridos','incômodo','dificuldade','nervoso','tristeza','sofrimento','caos',
        'agonia','desesperança','frustração','irritação','raiva','transtorno','angústia','desgaste','prejuízo','aflição',
        'desconforto','perturbação','abalo','stress','estresse','desespero','mágoa','culpa','vergonha','cansaço',
        'fadiga','exaustão','solidão','isolamento','medo','pânico','terror','pavor','aflição','sofrimento emocional','dano moral', 'dano estético'
    ]
    E de o resultado da sentença se foi favorável ou não ao autor da ação judicial., caso não tenha sentença apenas diga que não tem sentença.
        --- ESTRUTURA DE SAÍDA JSON ---
        
        Extraia as seguintes informações no formato JSON:
        1. sumario_fatos_Autor: Resumo detalhado (e anonimizado) focado na PETIÇÃO INICIAL, conforme a Regra 2.
        2. sumario_fatos_Réu: Resumo detalhado (e anonimizado) focado na CONTESTAÇÂO conforme a Regra 2.
        3. sumario_fatos_completo: Resumo detalhado (e anonimizado) do processo completo, integrando ambos os lados.
        4. tipo_falha_alegada: A categoria da falha (e.g., Sobrecarga, Curto-circuito, Falha de Equipamento, Indefinida).
        5. pericia_realizada: Se uma perícia técnica foi realizada (Sim/Não).
        6. parecer_pericia: Se a perícia foi favorável ao autor ou réu (Autor/Réu/Indefinido).
        7. dispositivo_sentenca: O resultado judicial (e.g., Improcedência, Dano Material).
        8. termos_tecnicos_autor: Todos termos técnicos relevantes (e não anonimizados) utilizados pelo réu. Colocar entre parenteses a quantidade que aparecem nos documentos do autor.
        9. termos_emocionais_autor: Todos termos argumentativos emocionais relevantes (e não anonimizados) utilizados pelo autor. Colocar entre parenteses a quantidade que aparecem nos documentos do autor.
        10. termos_tecnicos_réu: Todos termos técnicos relevantes (e não anonimizados) utilizados pelo réu. Colocar entre parenteses a quantidade que aparecem nos documentos do réu.
        11. termos_emocionais_réu: Todos termos argumentativos emocionais relevantes (e não anonimizados) utilizados pelo réu. Colocar entre parenteses a quantidade que aparecem nos documentos do réu.

        """
        
        # O modelo Gemini 2.5 Pro processa o prompt e o arquivo
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        # 3. Retorna o JSON processado
        return json.loads(response.text)

    except Exception as e:
        print(f"❌ Erro na chamada do Gemini: {e}")
        return None
        
    finally:
        # 4. Limpeza: Deleta o arquivo temporário do servidor do Gemini
        if uploaded_file:
            client.files.delete(name=uploaded_file.name)
            print("  -> Arquivo temporário deletado do servidor Gemini.")