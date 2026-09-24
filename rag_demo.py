"""Demo script for building RAG index and running a sample query.

Usage:
    python rag_demo.py build ./data/anon ./index_dir
    python rag_demo.py query ./index_dir "Como ocorreu a falha?" --top_k 5
"""
import sys
import argparse
from pathlib import Path

from src.rag import build_faiss_index, query_index, reduce_embeddings_for_plot


def cmd_build(args):
    print('Building index...')
    out = build_faiss_index(args.data_dir, args.index_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f'Index built at: {out}')


def cmd_query(args):
    print('Querying index...')
    results = query_index(args.index_dir, args.query, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        print(f"\nResult {i} | score={r['score']:.4f} | doc={r['metadata']['doc']}")
        print(r['metadata']['text'][:400])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()

    p_build = sub.add_parser('build')
    p_build.add_argument('data_dir')
    p_build.add_argument('index_dir')
    p_build.add_argument('--chunk_size', type=int, default=200)
    p_build.add_argument('--overlap', type=int, default=50)
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser('query')
    p_query.add_argument('index_dir')
    p_query.add_argument('query')
    p_query.add_argument('--top_k', type=int, default=5)
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    args.func(args)
