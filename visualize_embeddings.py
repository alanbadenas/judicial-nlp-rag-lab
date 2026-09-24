"""visualize_embeddings.py
Simple utility to load a .npy embedding file, print summary info and plot 2D projection.
Usage:
    python visualize_embeddings.py embeddings_single_process.npy
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

try:
    import umap
except Exception:
    umap = None


def main(path_str):
    path = Path(path_str)
    if not path.exists():
        print(f"File not found: {path}")
        return

    arr = np.load(path)
    print(f"Loaded: {path}")
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")
    print("Sample rows:")
    print(arr[:5])

    # Choose reducer
    reducer = None
    if umap is not None:
        try:
            reducer = umap.UMAP(n_components=2)
            emb2 = reducer.fit_transform(arr)
            print("Used UMAP for reduction")
        except Exception as e:
            print(f"UMAP failed: {e}")
            reducer = None

    if reducer is None and PCA is not None:
        try:
            reducer = PCA(n_components=2)
            emb2 = reducer.fit_transform(arr)
            print("Used PCA for reduction")
        except Exception as e:
            print(f"PCA failed: {e}")
            return

    if reducer is None:
        print("No reducer available (install umap-learn or scikit-learn). Skipping plot.")
        return

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(emb2[:,0], emb2[:,1], s=30, cmap='Spectral')
    plt.title(f"2D projection of {path.name}")
    out_png = path.with_suffix('.png')
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_embeddings.py <file.npy>")
    else:
        main(sys.argv[1])
