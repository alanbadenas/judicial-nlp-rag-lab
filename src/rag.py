"""RAG utilities: build FAISS index from text files and run queries.

Usage:
    from src.rag import build_faiss_index, query_index

Functions:
    build_faiss_index(data_dir, index_dir, model_name, chunk_size, overlap)
    query_index(index_dir, query, top_k)

Saves:
    index_dir/index.faiss  (faiss index)
    index_dir/embeddings.npy
    index_dir/metadata.json
"""
from pathlib import Path
import json
import os
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

try:
    import umap
except Exception:
    umap = None


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """Chunk text by words into overlapping chunks.
    chunk_size and overlap are measured in words.
    Returns list of (start_word_idx, end_word_idx, chunk_text).
    """
    words = text.split()
    chunks = []
    if chunk_size <= 0:
        return [(0, len(words), text)]
    i = 0
    n = len(words)
    while i < n:
        start = i
        end = min(i + chunk_size, n)
        chunk = " ".join(words[start:end])
        chunks.append((start, end, chunk))
        if end == n:
            break
        i = end - overlap
    return chunks


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_faiss_index(data_dir: str, index_dir: str, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', chunk_size: int = 200, overlap: int = 50):
    """Builds FAISS index from .txt files in data_dir. Saves index+metadata to index_dir.

    Returns path to index_dir.
    """
    data_path = Path(data_dir)
    index_path = Path(index_dir)
    ensure_dir(index_path)

    if SentenceTransformer is None:
        raise ImportError('sentence-transformers is required. Install with: pip install sentence-transformers')
    if faiss is None:
        raise ImportError('faiss is required. Install faiss-cpu or faiss-gpu')

    model = SentenceTransformer(model_name)

    embeddings = []
    metadata = []

    file_list = sorted([p for p in data_path.iterdir() if p.suffix.lower() == '.txt'])
    if not file_list:
        raise FileNotFoundError(f'No .txt files found in {data_dir}')

    doc_id = 0
    for p in file_list:
        text = p.read_text(encoding='utf-8', errors='ignore')
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for start, end, chunk in chunks:
            embed = model.encode(chunk)
            embeddings.append(embed)
            metadata.append({
                'doc': p.name,
                'doc_path': str(p.relative_to(data_path)),
                'start_word': int(start),
                'end_word': int(end),
                'text': chunk
            })
        doc_id += 1

    embeddings = np.vstack(embeddings).astype('float32')

    # normalize for cosine similarity with inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path / 'index.faiss'))
    np.save(index_path / 'embeddings.npy', embeddings)
    with open(index_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return str(index_path)


def query_index(index_dir: str, query: str, top_k: int = 5, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
    """Query the saved index and return top_k results with scores and metadata."""
    index_path = Path(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(index_dir)
    if SentenceTransformer is None:
        raise ImportError('sentence-transformers is required. Install with: pip install sentence-transformers')
    if faiss is None:
        raise ImportError('faiss is required. Install faiss-cpu or faiss-gpu')

    model = SentenceTransformer(model_name)
    index = faiss.read_index(str(index_path / 'index.faiss'))
    embeddings = np.load(index_path / 'embeddings.npy')
    with open(index_path / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    q_emb = model.encode(query).astype('float32')
    faiss.normalize_L2(q_emb.reshape(1, -1))
    D, I = index.search(q_emb.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if int(idx) < 0:
            continue
        meta = metadata[int(idx)]
        results.append({'score': float(score), 'metadata': meta})
    return results


def reduce_embeddings_for_plot(index_dir: str, n_neighbors: int = 15):
    """Optional: returns 2D coordinates for embeddings using UMAP or PCA fallback."""
    index_path = Path(index_dir)
    embeddings = np.load(index_path / 'embeddings.npy')
    if umap is not None:
        reducer = umap.UMAP(n_components=2)
        coords = reducer.fit_transform(embeddings)
        return coords
    else:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)
            return coords
        except Exception:
            return None
