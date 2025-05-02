import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

# 設定
CACHE_DIR = Path("cache")
EMBEDDING_CACHE = CACHE_DIR / "embeddings.pkl"
TOP_K = 1000

def batch_get_embeddings(texts: List[str], cache_key: str) -> np.ndarray:
    """テキストのリストの埋め込みを取得（キャッシュ機能付き）"""
    cache_path = EMBEDDING_CACHE.parent / f"embeddings_{cache_key}.pkl"
    
    if cache_path.exists():
        print(f"埋め込みをキャッシュから読み込み中: {cache_key}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print(f"埋め込みを計算中: {cache_key}")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings_array = np.array(embeddings.embed_documents(texts))
    
    # キャッシュに保存
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings_array, f)
    
    return embeddings_array

def compute_recall_at_k(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    positive_indices: List[List[int]],
    k: int = 100
) -> float:
    """Recall@Kを計算"""
    # コサイン類似度を計算
    similarities = query_embeddings @ corpus_embeddings.T
    
    # 各クエリについて上位k件を取得
    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
    
    # Recall@Kを計算
    total_recall = 0
    for i, (top_indices, pos_indices) in enumerate(zip(top_k_indices, positive_indices)):
        # 正解文書が上位k件に含まれている数を計算
        hits = sum(1 for pos_idx in pos_indices if pos_idx in top_indices)
        recall = hits / len(pos_indices)
        total_recall += recall
    
    return total_recall / len(query_embeddings)

def main():
    # データセットの読み込み
    print("データセットを読み込み中...")
    with open(CACHE_DIR / "sample_queries_100.pkl", "rb") as f:
        queries = pickle.load(f)
    with open(CACHE_DIR / "sample_corpus_10000.pkl", "rb") as f:
        corpus = pickle.load(f)
    
    # クエリと文書のテキストを抽出
    query_texts = [item["query"] for item in queries]
    corpus_texts = [item["text"] for item in corpus]
    
    # 正解文書のインデックスを取得
    positive_indices = []
    corpus_docids = [item["docid"] for item in corpus]
    for query in queries:
        pos_docids = [p["docid"] for p in query["positive_passages"]]
        pos_indices = [
            i for i, docid in enumerate(corpus_docids)
            if docid in pos_docids
        ]
        if not pos_indices:
            print(f"警告: クエリ '{query['query'][:50]}...' の正解文書が見つかりません")
        positive_indices.append(pos_indices)
    
    # 埋め込みを取得
    query_embeddings = batch_get_embeddings(query_texts, "queries")
    corpus_embeddings = batch_get_embeddings(corpus_texts, "corpus")
    
    # Recall@100を計算
    recall = compute_recall_at_k(
        query_embeddings, 
        corpus_embeddings, 
        positive_indices,
        k=TOP_K
    )
    
    print(f"\nRecall@{TOP_K}: {recall:.3f}")

if __name__ == "__main__":
    main() 