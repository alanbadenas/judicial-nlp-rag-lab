# src/pln_module.py

import numpy as np
from typing import Union
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import umap
import hdbscan

# ————— Helpers —————
class IdentityUMAP:
    """Transformador compatível com UMAP que não altera os embeddings."""
    def fit(self, X, y=None): return self
    def fit_transform(self, X, y=None): return X
    def transform(self, X): return X

def generate_embeddings(documents: list, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2') -> np.ndarray:
    """Gera S-BERT embeddings."""
    model = SentenceTransformer(model_name)
    emb = model.encode(documents, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb

def run_umap_reduction(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """UMAP seguro para N pequeno; fallback PCA."""
    X = np.asarray(embeddings)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]
    if N <= 1:
        return np.c_[np.zeros(N), np.zeros(N)][:, :max(1, n_components)]

    n_neighbors = max(2, min(15, N - 1))
    n_comp = max(1, min(n_components, N - 1))
    try:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_comp,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA
        n_comp = max(1, min(n_comp, min(N, X.shape[1])))
        return PCA(n_components=n_comp).fit_transform(X)

def run_bertopic(
    documents: list,
    embeddings: np.ndarray,
    use_umap: bool = True,
    umap_n_components: int = 4,
    umap_neighbors: int = 3,
    min_topic_size: Union[str, int] = "auto",
):
    """
    Executa BERTopic. Se use_umap=False, injeta um UMAP identidade (sem redução).
    Para N pequeno, ajusta parâmetros para evitar erros espectrais.
    Retorna o get_topic_info como dict.
    """
    from collections import Counter
    try:
        from .utils import simple_preprocess
    except ImportError:
        from utils import simple_preprocess

    docs = [d or "" for d in documents]
    X = np.asarray(embeddings)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Remove NaN/Inf
    if np.isnan(X).any() or np.isinf(X).any():
        mask = ~((np.isnan(X)).any(axis=1) | (np.isinf(X)).any(axis=1))
        X = X[mask]
        docs = [d for i, d in enumerate(docs) if mask[i]]

    N = len(docs)
    if N == 0 or X.size == 0:
        return {"Topic":[0], "Count":[0], "Name":["empty"], "Representation":[[]]}

    if X.shape[0] != N:
        m = min(N, X.shape[0])
        docs, X, N = docs[:m], X[:m], m

    # Fallback simples para N=1
    if N < 2:
        words = simple_preprocess(docs[0]).split()
        top = [w for w, _ in Counter(words).most_common(10)]
        return {"Topic":[0], "Count":[len(words)], "Name":["-1_unico_documento"], "Representation":[top]}

    # min_topic_size dinâmico
    mts = max(2, min(10, max(2, N // 5))) if min_topic_size == "auto" else int(min_topic_size)

    # Define modelo UMAP: real (seguro) ou identidade
    if use_umap and N >= 3:
        n_neighbors = max(2, min(umap_neighbors, N - 1))
        n_comp = max(2, min(umap_n_components, N - 1))
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_comp,
            metric="cosine",
            random_state=42,
        )
    else:
        umap_model = IdentityUMAP()  # não reduz

    # HDBSCAN configurado
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min(mts, N)),
        min_samples=None,
        metric="euclidean",
    )

    def _fit_with(umap_mdl):
        model = BERTopic(
            language="portuguese",
            umap_model=umap_mdl,          # IdentityUMAP → sem redução
            hdbscan_model=hdbscan_model,
            embedding_model=None,         # usando embeddings externos
            calculate_probabilities=False,
            low_memory=True,
            # reduce_outliers=False,        # evita passos extras em N pequeno
            verbose=False,
        )
        topics, _ = model.fit_transform(docs, X)
        info = model.get_topic_info()
        return info.to_dict(orient="list")

    # Tenta com o UMAP definido acima
    try:
        return _fit_with(umap_model)
    except Exception as e1:
        print(f"⚠️ BERTopic falhou com umap_model={type(umap_model).__name__}: {e1}. Tentando IdentityUMAP...")
        try:
            return _fit_with(IdentityUMAP())
        except Exception as e2:
            print(f"⚠️ BERTopic (IdentityUMAP) falhou: {e2}. Fallback de frequência.")
            bag = []
            for d in docs:
                bag.extend(simple_preprocess(d).split())
            top = [w for w, _ in Counter(bag).most_common(10)] or ["-"]
            return {"Topic":[0], "Count":[N], "Name":["-1_fallback_error"], "Representation":[top]}