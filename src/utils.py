# src/utils.py
import pytesseract
import re
from pypdf import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from anonymization_module import get_ner_pipeline, anonymize_text
import nltk
import json
from pathlib import Path
from typing import List, Set

# Garantir recursos NLTK (para evitar erro se for executado fora do Notebook)
def ensure_nltk_resources():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab/portuguese', 'punkt_tab')
    ]
    for res_path, res_name in resources:
        try:
            nltk.data.find(res_path)
        except LookupError:
            try:
                print(f"Baixando recurso NLTK: {res_name} ...")
                nltk.download(res_name)
            except Exception as e:
                print(f"Falha ao baixar {res_name}: {e}")


ensure_nltk_resources()

# Carrega stopwords (se falhar, usa conjunto vazio)
try:
    STOP_WORDS = set(stopwords.words('portuguese'))
except Exception:
    print("Aviso: não foi possível carregar stopwords do NLTK. Usando conjunto vazio.")
    STOP_WORDS = set()

try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    print("Alerta: Não foi possível definir o caminho do Tesseract. O OCR pode falhar.")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrai texto de um arquivo PDF."""
    # (Copie a função extract_text_from_pdf da Célula 2)
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Erro ao ler PDF: {e}. Verifique o caminho.")
    return text

def anonymize_regex(text: str) -> str:
    """Anonimiza padrões comuns via RegEx (CPF, N° Processo, Datas)."""
    # (Copie a função anonymize_regex da Célula 2)
    text = re.sub(r'\d{3}\.\d{3}\.\d{3}-\d{2}', '[ANON_CPF]', text)
    text = re.sub(r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}', '[ANON_NUM_PROCESSO]', text)
    text = re.sub(r'\d{2}/\d{2}/\d{4}', '[ANON_DATA]', text)
    text = re.sub(r'([A-Z][a-z]+)\s+([A-Z][a-z]+)', r'[ANON_NOME]', text) 
    return text

# Caminho para o arquivo que registrará os PDFs processados
PROCESSED_LOG_FILE = Path("./processed_files_log.json")

def get_files_to_process(data_dir: str = "./data") -> List[Path]: # Corrigido: List importado
    """
    Lista todos os arquivos PDF no diretório de dados que ainda não foram processados.
    """
    # ... (restante da função) ...
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Diretório de dados não encontrado: {data_dir}")
        return []

    all_pdfs = list(data_path.glob("*.pdf"))
    
    # Carrega o log de processados
    processed_files_log = load_processed_log()
    
    files_to_process = []
    for pdf_path in all_pdfs:
        # Usa o nome do arquivo para verificar se já foi processado
        if pdf_path.name not in processed_files_log:
            files_to_process.append(pdf_path)
        else:
            print(f"[*] Ignorando {pdf_path.name} (Já processado no log).")
            
    print(f"Encontrados {len(all_pdfs)} PDFs. {len(files_to_process)} a processar.")
    return files_to_process


def load_processed_log() -> Set[str]: # Corrigido: Set importado
    """Carrega o set de nomes de arquivos processados do log JSON."""
    if not PROCESSED_LOG_FILE.exists():
        return set()
    try:
        with open(PROCESSED_LOG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Garante que o log retorne um set de strings
            return set(data.get("processed_files", [])) 
    except:
        return set()

def update_processed_log(filename: str):
    """Adiciona o nome do arquivo processado ao log JSON."""
    log_data = load_processed_log()
    log_data.add(filename)
    
    # Salva de volta
    with open(PROCESSED_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump({"processed_files": list(log_data)}, f, indent=4)

def run_anonymization_pipeline(text: str) -> str:
    """Executa o pipeline completo de anonimização: LENERBR + RegEx + Normalização/Limpeza."""
    # 1. LENERBR (Reconhecimento de Entidades Nomeadas Jurídicas)
    ner_pipe = get_ner_pipeline()
    print("  -> Aplicando LENERBR para Anonimização Contextual...")
    text = anonymize_text(text, ner_pipe)
    
    # 2. RegEx (Padrões fixos que o modelo pode perder)
    print("  -> Aplicando RegEx para Padrões Numéricos...")
    text = anonymize_regex(text)
    
    return text

def simple_preprocess(text: str) -> str:
    """NOVA ORDEM: LENERBR -> RegEx -> Normalização/Limpeza."""
    
    # # 1. LENERBR (Reconhecimento de Entidades Nomeadas Jurídicas)
    # ner_pipe = get_ner_pipeline()
    # print("  -> Aplicando LENERBR para Anonimização Contextual...")
    # text = anonymize_text(text, ner_pipe)
    
    # # 2. RegEx (Padrões fixos que o modelo pode perder)
    # print("  -> Aplicando RegEx para Padrões Numéricos...")
    # text = anonymize_regex(text)
    
    # 3. Normalização e Limpeza
    # ... (Restante do código de normalização, lower, stopwords)
    text = text.lower()
    text = re.sub(r'[^a-záéíóúâêôãõç\s\[\]<>]', ' ', text) # Manter as tags de anonimização!
    # Tenta tokenização com NLTK; se os recursos não estiverem disponíveis,
    # usa um tokenizador fallback simples (split por espaço)
    try:
        tokens = word_tokenize(text, language='portuguese')
    except LookupError as e:
        print(f"Aviso: recurso NLTK ausente ({e}). Usando tokenização simples.")
        tokens = text.split()
    except Exception as e:
        print(f"Aviso: erro na tokenização NLTK ({e}). Usando tokenização simples.")
        tokens = text.split()

    tokens_filtered = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    
    return " ".join(tokens_filtered)

def save_pdf_as_txt(pdf_path: str, out_dir: str = "./exports/txt") -> str:
    """Extrai texto do PDF e salva como .txt no out_dir. Retorna o caminho salvo."""
    text = extract_text_from_pdf(pdf_path)
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / (Path(pdf_path).stem + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text or "")
    return str(out_path)

def anonymize_txt_folder(in_dir: str = "./exports/txt",
                         out_dir: str = "./exports/txt_anon",
                         method: str = "pipeline") -> str:
    """
    Anonimiza todos os .txt de in_dir e salva em out_dir.
    method: "pipeline" -> usa run_anonymization_pipeline (NER+Regex)
            "regex"    -> usa apenas anonymize_regex (sem dependências extras)
    Retorna o caminho do manifest CSV.
    """
    from pathlib import Path
    import csv

    in_p = Path(in_dir); out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("./exports/metrics"); metrics_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_p.glob("*.txt"))
    if not files:
        print(f"⚠️ Nenhum .txt em {in_dir}")
        return ""

    manifest = metrics_dir / "txt_anonymization_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_in","file_out","method_used","len_in","len_out","status","error"])
        for fp in files:
            try:
                raw = fp.read_text(encoding="utf-8", errors="ignore")
                if method == "regex":
                    out_txt = anonymize_regex(raw)
                    used = "regex"
                else:
                    try:
                        out_txt = run_anonymization_pipeline(raw)
                        used = "pipeline"
                    except Exception as e:
                        # fallback para regex
                        print(f"⚠️ Pipeline falhou em {fp.name}: {e}. Usando regex.")
                        out_txt = anonymize_regex(raw)
                        used = "regex(fallback)"

                out_fp = out_p / fp.name
                out_fp.write_text(out_txt or "", encoding="utf-8")
                w.writerow([str(fp), str(out_fp), used, len(raw), len(out_txt or ""), "ok", ""])
            except Exception as e:
                w.writerow([str(fp), "", method, 0, 0, "fail", str(e)])
                print(f"❌ Falha ao anonimizar {fp.name}: {e}")

    print(f"💾 Manifest salvo: {manifest}")
    print(f"🖹 TXT anonimizados em: {out_p}")
    return str(manifest)
# Funções de IO podem ser adicionadas aqui (ex: salvar/carregar JSON)