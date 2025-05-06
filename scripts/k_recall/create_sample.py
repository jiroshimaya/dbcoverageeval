import os
import pickle
import time
from pathlib import Path
from typing import Any, Tuple

import datasets
import numpy as np
import tiktoken
import torch
import tqdm

# 保存用のディレクトリを作成
CACHE_DIR = Path("cache/k_recall")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def get_openai_embeddings(
    texts: list[str],
    *,
    model_name: str = "text-embedding-3-small",
    dimensions: int = 1536,
    max_tokens_per_batch: int = 2000000,
    max_chars_per_text: int = 2000,
    interval_seconds: float = 60,
) -> np.ndarray:
    from langchain_openai import OpenAIEmbeddings

    if model_name.startswith("text-embedding-3"):
        client = OpenAIEmbeddings(model=model_name, dimensions=dimensions)
    else:
        client = OpenAIEmbeddings(model=model_name)

    enc = tiktoken.encoding_for_model(model_name)

    texts = [text.replace("\n", " ")[:max_chars_per_text] for text in texts]
    all_embeddings = []

    token_count = 0
    current_texts = []
    is_first = True
    pbar = tqdm.tqdm(total=len(texts), desc="Embedding texts")
    while texts:
        text = texts.pop(0)
        if token_count + len(text) > max_tokens_per_batch:
            if not is_first:
                time.sleep(interval_seconds)
            embs = client.embed_documents(current_texts)
            is_first = False
            all_embeddings += embs
            pbar.update(len(current_texts))
            current_texts = [text]
            token_count = len(enc.encode(text))
        else:
            current_texts.append(text)
            token_count += len(enc.encode(text))
    if current_texts:
        embs = client.embed_documents(current_texts)
        all_embeddings += embs
        pbar.update(len(current_texts))
    pbar.close()
    return np.array(all_embeddings)


def get_huggingface_embeddings(
    texts: list[str],
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        "intfloat/multilingual-e5-base",
        device=device
    )
    embs = model.encode(texts,
                        batch_size=128,       # 16 GB VRAM なら目安は 96–128
                        normalize_embeddings=True,
                        show_progress_bar=True,
                        )

    return np.array(embs)


def load_or_create_datasets() -> Tuple[datasets.Dataset, datasets.Dataset]:
    corpus_path = CACHE_DIR / "all_corpus.pkl"
    query_path = CACHE_DIR / "all_queries.pkl"

    # キャッシュされたデータセットがある場合はそれを読み込む
    if corpus_path.exists() and query_path.exists():
        print("キャッシュされたデータセットを読み込みます...")
        with open(corpus_path, "rb") as f:
            full_corpus = pickle.load(f)
        with open(query_path, "rb") as f:
            full_ds = pickle.load(f)
    else:
        print("データセットをダウンロードします...")
        # クエリデータセットを取得（全件）
        full_ds = datasets.load_dataset(
            "miracl/miracl", "ja", token=os.environ["HF_ACCESS_TOKEN"], split="dev"
        )
        full_corpus = datasets.load_dataset("miracl/miracl-corpus", "ja", split="train")
        print("データセットをキャッシュに保存します...")
        with open(corpus_path, "wb") as f:
            pickle.dump(full_corpus, f)
        with open(query_path, "wb") as f:
            pickle.dump(full_ds, f)

    return full_corpus, full_ds


def load_or_create_embeddings(
    full_corpus: datasets.Dataset, full_ds: datasets.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    corpus_path = CACHE_DIR / "corpus_embeddings.pkl"
    query_path = CACHE_DIR / "queries_embeddings.pkl"
    batch_size = 100000

    # --- Corpus Embeddings ---
    if corpus_path.exists():
        print("キャッシュされた埋め込みを読み込みます... (corpus)")
        with open(corpus_path, "rb") as f:
            corpus_embeddings = pickle.load(f)
    else:
        print("埋め込みを計算します... (corpus)")
        corpus_texts = [f'passage: {item["title"]} {item["text"]}' for item in full_corpus]
        num_batches = (len(corpus_texts) + batch_size - 1) // batch_size
        batch_emb_files = []
        for i in range(num_batches):
            batch_file = CACHE_DIR / f"corpus_embeddings_part_{i}.pkl"
            batch_emb_files.append(batch_file)
            if batch_file.exists():
                print(f"バッチ {i}/{num_batches} は既に存在します。スキップします。")
                continue
            print(f"バッチ {i}/{num_batches} を計算中...")
            batch_texts = corpus_texts[i*batch_size:(i+1)*batch_size]
            batch_embs = get_huggingface_embeddings(batch_texts)
            with open(batch_file, "wb") as f:
                pickle.dump(batch_embs, f)
        # 全バッチを結合
        all_embs = []
        for batch_file in tqdm.tqdm(batch_emb_files, desc="load corpus embeddings"):
            with open(batch_file, "rb") as f:
                all_embs.append(pickle.load(f))
        corpus_embeddings = np.concatenate(all_embs, axis=0)
        with open(corpus_path, "wb") as f:
            pickle.dump(corpus_embeddings, f)

    # --- Query Embeddings ---
    if query_path.exists():
        print("キャッシュされた埋め込みを読み込みます... (query)")
        with open(query_path, "rb") as f:
            query_embeddings = pickle.load(f)
    else:
        print("埋め込みを計算します... (query)")
        query_texts = [f'query: {item["query"]}' for item in full_ds]
        query_embeddings = get_huggingface_embeddings(query_texts)
        with open(query_path, "wb") as f:
            pickle.dump(query_embeddings, f)

    return corpus_embeddings, query_embeddings


def main():
    # データセットを取得
    sample_corpus, sample_ds = load_or_create_datasets()
    corpus_embeddings, query_embeddings = load_or_create_embeddings(
        sample_corpus, sample_ds
    )

    # 正解文書の数を確認

    print(f"コーパスサイズ: {len(sample_corpus)}")
    print(f"クエリ数: {len(sample_ds)}")

    # サンプルデータの内容確認
    print("\nサンプルクエリ:")
    print(sample_ds[0]["query"])
    print("\n対応する正解文書:")
    print(sample_ds[0]["positive_passages"][0]["text"][:100])


if __name__ == "__main__":
    main()
