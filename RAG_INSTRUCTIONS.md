RAG-light instructions

Overview
--------
This project builds a simple Retrieval-Augmented-Generation (RAG) pipeline over anonymized legal texts.
It creates passage chunks, generates embeddings with sentence-transformers, indexes them with FAISS, and allows semantic queries.

Quick setup
-----------
1. Activate your venv (PowerShell):

    & .\venv\Scripts\Activate.ps1

2. Install required packages:

    pip install sentence-transformers faiss-cpu umap-learn scikit-learn

(If you don't need UMAP for plotting, at least install scikit-learn.)

Build index
-----------
Run (example):

    python rag_demo.py build ./data/anon ./index_dir --chunk_size 200 --overlap 50

This will:
- Read all `.txt` files in `./data/anon` (one doc per file),
- Chunk them into overlapping passages (200 words, 50 overlap by default),
- Encode passages with `paraphrase-multilingual-MiniLM-L12-v2`,
- Normalize embeddings for cosine search, create FAISS index and save:
    - `index_dir/index.faiss`
    - `index_dir/embeddings.npy`
    - `index_dir/metadata.json`

Query
-----
Once index is built, run:

    python rag_demo.py query ./index_dir "Como ocorreu a falha?" --top_k 5

This prints top-k passages with scores and the passage text.

Run Streamlit app
------------------
You can run an interactive Streamlit interface included in the project:

    streamlit run streamlit_rag_app.py

The app allows building the index (button), running queries, visualising the 2D projection and inspecting document-level statistics (technical/emotional term counts). Use the sidebar to point to your `index_dir` and `data/anon` paths.

Visualization (optional)
------------------------
You can reduce embeddings to 2D (UMAP/PCA) using `src.rag.reduce_embeddings_for_plot(index_dir)` from Python and plot with matplotlib.

Testing plan (recommended)
--------------------------
1. Prepare 10 representative queries you expect users to ask (technical and factual).
2. For each query, run the `query` command and record top-5 passages.
3. Compute precision@5 manually or with a small CSV of relevance judgments.
4. Evaluate privacy: check top returned passages for PII leakage; annotate if any PII remains.

Next steps to make it visual / interactive
----------------------------------------
- Add a Streamlit app that loads the index and provides an input box for queries, displays top passages and optional summarization.
- Integrate a simple summarizer (TextRank) to generate short answers from the top passages.

If you want, I can implement the Streamlit demo and add a notebook cell that runs build+query with interactive widgets.