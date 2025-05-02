import os
import pickle
import random
from pathlib import Path
from typing import Set, Tuple

import datasets

# 保存用のディレクトリを作成
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_positive_docids(ds) -> Set[str]:
    """データセットから正解文書のdocidを取得"""
    docids = set()
    for item in ds:
        for passage in item["positive_passages"]:
            docids.add(passage["docid"])  # "#" 以降を除去
    return docids

def load_or_create_datasets(corpus_size=10000, query_size=100) -> Tuple[datasets.Dataset, datasets.Dataset]:
    corpus_path = CACHE_DIR / f"sample_corpus_{corpus_size}.pkl"
    query_path = CACHE_DIR / f"sample_queries_{query_size}.pkl"
    
    # キャッシュされたデータセットがある場合はそれを読み込む
    if corpus_path.exists() and query_path.exists():
        print("キャッシュされたデータセットを読み込みます...")
        with open(corpus_path, "rb") as f:
            sample_corpus = pickle.load(f)
        with open(query_path, "rb") as f:
            sample_ds = pickle.load(f)
    else:
        print("データセットをダウンロードします...")
        # クエリデータセットを取得
        full_ds = datasets.load_dataset(
            "miracl/miracl", 
            "ja", 
            token=os.environ["HF_ACCESS_TOKEN"], 
            split="dev"
        )
        sample_ds = full_ds.select(range(query_size))
        
        # 正解文書のdocidを取得
        print("正解文書のdocidを取得します...")
        positive_docids = get_positive_docids(sample_ds)
        pos_size = len(positive_docids)
        
        print("コーパスをロードします...")
        full_corpus = datasets.load_dataset("miracl/miracl-corpus", "ja", split="train")
        
        # 初期サブセットを作成（corpus_size + 正解文書数*10）
        print("初期サブセットを作成します...")
        initial_subset_size = corpus_size + pos_size * 10
        initial_subset = full_corpus.select(random.sample(range(len(full_corpus)), initial_subset_size))
        
        # 初期サブセットから非正解文書のみを抽出
        print("非正解文書を抽出します...")
        neg_docs = [doc for doc in initial_subset if doc["docid"] not in positive_docids]
        
        # 必要な数の負例をサンプリング
        print("負例をサンプリングします...")
        neg_size = corpus_size - pos_size
        assert len(neg_docs) >= neg_size, f"十分な負例が得られませんでした。初期サブセットサイズを大きくしてください。(必要数: {neg_size}, 取得数: {len(neg_docs)})"
        
        sampled_neg_docs = random.sample(neg_docs, neg_size)
        
        # 最終的なコーパスを作成
        print("最終的なコーパスを作成します...")
        # 正解文書をfilterで取得
        pos_docs = full_corpus.filter(lambda x: x["docid"] in positive_docids)
        sample_corpus = datasets.Dataset.from_list(list(pos_docs) + sampled_neg_docs)
        
        # データセットを保存
        print("データセットをキャッシュに保存します...")
        with open(corpus_path, "wb") as f:
            pickle.dump(sample_corpus, f)
        with open(query_path, "wb") as f:
            pickle.dump(sample_ds, f)
    
    return sample_corpus, sample_ds

def main():
    # データセットを取得
    sample_corpus, sample_ds = load_or_create_datasets()
    
    # 正解文書の数を確認
    positive_docids = get_positive_docids(sample_ds)
    positive_docs = [doc for doc in sample_corpus if doc["docid"] in positive_docids]
    
    print(f"コーパスサイズ: {len(sample_corpus)}")
    print(f"クエリ数: {len(sample_ds)}")
    print(f"正解文書数: {len(positive_docs)}")
    print(f"負例文書数: {len(sample_corpus) - len(positive_docs)}")
    
    # サンプルデータの内容確認
    print("\nサンプルクエリ:")
    print(sample_ds[0]["query"])
    print("\n対応する正解文書:")
    print(sample_ds[0]["positive_passages"][0]["text"][:100])

if __name__ == "__main__":
    main()
