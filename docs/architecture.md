# Architecture

```mermaid
flowchart LR
    A[Local documents] --> B[Text extraction]
    B --> C[Deterministic identifier redaction]
    C --> D[Contextual NER anonymization]
    D --> E[Preprocessing]
    E --> F[Sentence-BERT embeddings]
    F --> G[UMAP / HDBSCAN / BERTopic]
    F --> H[FAISS vector index]
    H --> I[Semantic retrieval]
    G --> J[Exploratory analysis]
```

The public demo uses synthetic `.txt` files. Optional PDF and LLM-assisted paths are retained as academic experiments but should not be used with sensitive documents without a separate privacy/security review.
