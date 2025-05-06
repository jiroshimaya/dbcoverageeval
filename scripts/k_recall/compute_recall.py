import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
from scipy.spatial.distance import cdist

import numpy as np
from scipy.stats import hypergeom
from tqdm import tqdm


# 設定
CACHE_DIR = Path("cache/k_recall")
S = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
K = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000]

def compute_recall_for_size_and_k(
    ranks_list: list[np.ndarray],
    N: int,
    S: list[int] = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000],
    K: list[int] = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000]
) -> np.ndarray:
    # ranks_list: list of np.array (長さ R_q), クエリ数 = Q
    # 例: ranks_list[q] = np.array([12, 345, 7890])  # そのクエリの全 positive の順位
    # ranks_list = [...]            # ←あなたのデータをここに

    # クエリ数
    Q = len(ranks_list)
    recall = np.zeros((len(S), len(K)))

    for si, s in enumerate(S):
        for kj, k in enumerate(K):
            recall_q = np.zeros(Q)            # 各クエリの recall 期待値
            for q, ranks in enumerate(ranks_list):
                # 正例数
                R = len(ranks)
                # サブコーパスが positives 以下なら必ず全ヒット
                if s <= k or ranks.min() <= k or s <= R:
                    recall_q[q] = 1.0
                    continue

                n = s - R                     # 抽出する負例数
                H = ranks - 1
                L = N - ranks
                # P_i = P(X ≤ k-1)
                P = hypergeom.cdf(k - 1, H + L, H, n)
                recall_q[q] = P.mean()        # (1/R) Σ P_i
            recall[si, kj] = recall_q.mean()  # クエリ平均
    return recall

# recall 行列 :  shape = (len(S), len(K))


def load_or_compute_ranks_list(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    positive_indices: list[list[int]],
    save_path: str = CACHE_DIR / "ranks_list.pkl",
    save_every: int = 10,
) -> list[np.ndarray]:
    ranks_list = []
    start_idx = 0
    # 途中までのキャッシュがあればロード
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            ranks_list = pickle.load(f)
        start_idx = len(ranks_list)
        print(f"キャッシュから{start_idx}件分をロードしました。続きから計算します。")
    else:
        print("キャッシュが見つかりません。最初から計算します。")
    # メモリを節約するためにクエリごとに類似度計算
    for i in tqdm(range(start_idx, len(query_embeddings))):
        q_emb = query_embeddings[i]
        pos_idx = positive_indices[i]
        sim = cdist([q_emb], corpus_embeddings, metric="cosine")[0]
        sorted_idx = np.argsort(sim)
        ranks = np.where(np.isin(sorted_idx, pos_idx))[0]
        ranks_list.append(ranks)
        # save_everyごとにキャッシュ保存
        if ((i + 1) % save_every == 0 or (i + 1) == len(query_embeddings)):
            with open(save_path, "wb") as f:
                pickle.dump(ranks_list, f)
            print(f"{i + 1}件分を{save_path}に保存しました。")
    return ranks_list

def main():
    print("埋め込みを読み込み中...")
    with open(CACHE_DIR / "corpus_embeddings.pkl", "rb") as f:
        corpus_embeddings = pickle.load(f)
    with open(CACHE_DIR / "queries_embeddings.pkl", "rb") as f:
        query_embeddings = pickle.load(f)
    
    positive_indices_path = CACHE_DIR / "positive_indices.json"
    if positive_indices_path.exists():
        print("正解文書のインデックスを読み込み中...")
        with open(positive_indices_path, "rb") as f:
            positive_indices = json.load(f)
    else:
        with open(CACHE_DIR / "all_queries.pkl", "rb") as f:
            queries = pickle.load(f)
        with open(CACHE_DIR / "all_corpus.pkl", "rb") as f:
            corpus = pickle.load(f)
            
        
        # 正解文書のインデックスを取得
        print("正解文書のインデックスを取得中...")
        positive_indices = []
        corpus_docids = {item["docid"]: i for i, item in enumerate(corpus)}
        for query in queries:
            pos_docids = [p["docid"] for p in query["positive_passages"]]
            pos_indices = [
                corpus_docids[docid] for docid in pos_docids
            ]
            if not pos_indices:
                print(f"警告: クエリ '{query['query'][:50]}...' の正解文書が見つかりません")
            positive_indices.append(pos_indices)
        
        # メモリ解放
        del queries, corpus
        import gc
        gc.collect()
    
    print("ランクリストを計算中...")
    ranks_list = load_or_compute_ranks_list(query_embeddings, corpus_embeddings, positive_indices)
    print("Recall@Kを計算中...")
    recall = compute_recall_for_size_and_k(ranks_list, len(corpus_embeddings), S, K)
    print(f"\nRecall@{TOP_K}: {recall:.3f}")
    
    np.savetxt(CACHE_DIR / "recall.csv", recall, delimiter=",")
    print(f"recallを {CACHE_DIR / 'recall.csv'} に保存しました")

if __name__ == "__main__":
    main() 