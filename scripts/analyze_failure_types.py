import json
from pathlib import Path
import unicodedata
import re
from collections import Counter, defaultdict

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

INPUT = Path("IntroCD/04 - Pesquisa/data/final_dataset.json")
OUT_DIR = Path("exports/analysis_failures"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(s: str) -> str:
    if not s: return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# simples mapeamento por palavras-chave para agrupar rótulos semelhantes
MAP_RULES = [
    (["sobretens", "oscil", "sobretensão", "oscila"], "Sobretensão / Oscilação"),
    (["descarga", "raio", "descarga atmos", "descarga elétrica"], "Descarga / Atmosférica"),
    (["falha de equipamento", "vício", "vício oculto", "equipamento", "peça"], "Falha de Equipamento / Vício"),
    (["inadimplemento", "inadimplemento contratual", "inadimpl", "recisao", "rescisão"], "Inadimplemento Contratual"),
    (["concorrência", "pirataria", "propriedade intelectual", "violação", "importação paralela"], "Concorrência / Pirataria"),
    (["medidor", "adulteração", "desvio", "bobina", "lacre"], "Medidor / Desvio / Adulteração"),
    (["curto", "curto-circuito"], "Curto-circuito / Sobretensão"),
    (["inversão de fluxo", "inversão de fluxo de potência"], "Inversão de Fluxo (Geração)"),
    (["servidão", "servidão administrativa"], "Servidão / Servidão Administrativa"),
    (["fraude", "furto de energia", "desvio de energia"], "Fraude / Desvio de Energia"),
]

def map_label(raw: str):
    s = normalize_text(raw)
    if not s:
        return "MISSING"
    for keys, lab in MAP_RULES:
        for k in keys:
            if k in s:
                return lab
    # fallback: first 5 words capitalized
    return ("OTHER: " + " ".join(s.split()[:5])).title()

def top_terms_per_group(texts, n=15):
    vec = CountVectorizer(max_features=2000, stop_words="portuguese", ngram_range=(1,2))
    X = vec.fit_transform(texts)
    fn = vec.get_feature_names_out()
    sums = X.sum(axis=0).A1
    idx = sums.argsort()[::-1][:n]
    return [fn[i] for i in idx]

def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    df["tipo_raw"] = df.get("tipo_falha_alegada", "").fillna("").astype(str)
    df["tipo_norm"] = df["tipo_raw"].apply(map_label)
    df["text"] = (df.get("sumario_fatos_completo") or "").astype(str).fillna("").str.replace(r"\[ANON_[^\]]+\]", " ", regex=True)

    # counts
    counts = df["tipo_norm"].value_counts().to_dict()
    print("\n== Counts (tipo_falha_alegada normalized) ==")
    for k, v in counts.items():
        print(f" {v:>2}  {k}")

    # per-group diagnostics
    summary = {}
    for lab, sub in df.groupby("tipo_norm"):
        texts = sub["text"].tolist()
        summary[lab] = {
            "n_docs": int(len(sub)),
            "sample_docs": sub.index[:5].tolist(),
            "avg_tokens": float(sub["text"].str.split().map(len).mean()) if len(sub)>0 else 0,
            "pericia_counts": sub["pericia_realizada"].value_counts().to_dict() if "pericia_realizada" in df.columns else {}
        }
        # top terms
        try:
            summary[lab]["top_terms"] = top_terms_per_group(texts, n=15)
        except Exception:
            summary[lab]["top_terms"] = []

    # save outputs
    pd.Series(counts).to_csv(OUT_DIR / "failure_type_counts.csv", header=["count"])
    pd.DataFrame.from_dict(summary, orient="index").to_json(OUT_DIR / "failure_type_summary.json", force_ascii=False, indent=2)

    print(f"\nSaved: {OUT_DIR / 'failure_type_counts.csv'} and {OUT_DIR / 'failure_type_summary.json'}")
    print("\n== Recommendation ==")
    total = len(df)
    n_labels = len(counts)
    print(f" total docs: {total}, distinct normalized labels: {n_labels}")
    print(" - If many labels have <10 examples, GROUP similar labels (e.g. merge 'Sobretensão' + 'Curto-circuito' etc.)")
    print(" - For classification target: 30 docs is small; prefer 2-4 coarse classes or collect more labeled examples.")
    print(" - Use the JSON output to pick candidate merges and manual relabeling.")

if __name__ == "__main__":
    main()