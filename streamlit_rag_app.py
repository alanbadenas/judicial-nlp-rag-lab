import streamlit as st
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os

st.set_page_config(layout='wide', page_title='RAG - Projeto_DataJud')

st.title('RAG-light Demo — Projeto_DataJud')

st.sidebar.header('Config')
index_dir = st.sidebar.text_input('Index directory', './index_dir')
data_dir = st.sidebar.text_input('Data directory (anon)', './data/anon')
model_name = st.sidebar.text_input('Embedding model', 'paraphrase-multilingual-MiniLM-L12-v2')
chunk_size = st.sidebar.number_input('Chunk size (words)', value=200, step=50)
overlap = st.sidebar.number_input('Overlap (words)', value=50, step=10)

from src.rag import build_faiss_index, query_index, reduce_embeddings_for_plot
from src.analysis_pipeline import load_documents, build_dataframe

# Build index button
if st.sidebar.button('Build index'):
    with st.spinner('Building FAISS index — this may take a while...'):
        try:
            out = build_faiss_index(data_dir, index_dir, model_name=model_name, chunk_size=chunk_size, overlap=overlap)
            st.success(f'Index built at {out}')
        except Exception as e:
            st.error(f'Error building index: {e}')

st.sidebar.markdown('---')

query = st.text_input('Query', '')
top_k = st.slider('Top k', 1, 20, 5)

if st.button('Search') and query.strip():
    with st.spinner('Searching...'):
        try:
            results = query_index(index_dir, query, top_k=top_k, model_name=model_name)
        except Exception as e:
            st.error(f'Query failed: {e}')
            results = []

    # Load metadata if available
    meta_path = Path(index_dir) / 'metadata.json'
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = None

    # Try to load dataset mapping (dispositivo_sentenca) from common files
    dataset_map = {}
    possible_dataset_files = [
        Path(data_dir).parent / 'dataset_inicial.json',
        Path(data_dir).parent / 'final_dataset.json',
        Path(data_dir) / 'dataset_inicial.json',
        Path(data_dir) / 'final_dataset.json',
        Path('dataset_inicial.json'),
        Path('final_dataset.json')
    ]
    for p in possible_dataset_files:
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    dd = json.load(f)
                # Expecting list of records or dict mapping filenames -> record
                if isinstance(dd, dict):
                    for k, v in dd.items():
                        # try to extract dispositivo_sentenca if present
                        if isinstance(v, dict) and 'dispositivo_sentenca' in v:
                            dataset_map[k] = v.get('dispositivo_sentenca')
                elif isinstance(dd, list):
                    for item in dd:
                        if isinstance(item, dict):
                            # try common keys: 'file', 'doc', 'doc_id', 'nome_arquivo'
                            key = item.get('file') or item.get('doc') or item.get('doc_id') or item.get('nome_arquivo')
                            if key and 'dispositivo_sentenca' in item:
                                dataset_map[key] = item.get('dispositivo_sentenca')
            except Exception:
                # ignore parse errors and continue
                pass

    # Display results
    cols = st.columns([3,1])
    with cols[0]:
        st.subheader('Top passages')
        for i, r in enumerate(results, 1):
            md = r['metadata']
            score = r['score']
            docname = md.get('doc')
            st.markdown(f"**{i}. doc={docname} — score={score:.4f}**")
            st.write(md.get('text'))
            # show dispositivo_sentenca if available
            disp = None
            if docname in dataset_map:
                disp = dataset_map[docname]
            else:
                # try basename match
                if docname:
                    bn = Path(docname).name
                    disp = dataset_map.get(bn)
            if disp:
                st.info(f"dispositivo_sentenca: {disp}")
            st.markdown('---')
    with cols[1]:
        st.subheader('Document info')
        # Build df of docs via analysis pipeline
        try:
            docs = load_documents(data_dir)
            df, topics_words = build_dataframe(docs)
            # try to attach dispositivo_sentenca to df if we have mapping
            if not df.empty and dataset_map:
                # map on doc_id or filename-like cols
                if 'doc_id' in df.columns:
                    df['dispositivo_sentenca'] = df['doc_id'].map(lambda x: dataset_map.get(x) or dataset_map.get(Path(str(x)).name))
                elif 'filename' in df.columns:
                    df['dispositivo_sentenca'] = df['filename'].map(lambda x: dataset_map.get(x) or dataset_map.get(Path(str(x)).name))
            cols_to_show = [c for c in ['doc_id','technical_count','emotional_count','technical_ratio','emotional_ratio','cluster','dominant_topic','dispositivo_sentenca'] if c in df.columns]
            st.dataframe(df[cols_to_show])
        except Exception as e:
            st.info('Could not compute document statistics: ' + str(e))

    # Visualization: embeddings 2D
    with st.expander('Show 2D projection of embeddings'):
        coords = reduce_embeddings_for_plot(index_dir)
        if coords is None:
            st.info('No reducer available (install umap-learn or scikit-learn)')
        else:
            import plotly.express as px
            emb_meta = []
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                emb_meta = [m.get('doc') for m in meta]
            df_coords = pd.DataFrame(coords, columns=['x','y'])
            if emb_meta:
                df_coords['doc'] = emb_meta
            fig = px.scatter(df_coords, x='x', y='y', hover_data=['doc'], title='Embeddings 2D')
            st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown('---')
st.sidebar.markdown('Instructions: build index first or run `python rag_demo.py build` in shell. Then query here.')
