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
CACHE_DIR.mkdir(exist_ok=True)


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
    *,
    model_name: str = "intfloat/multilingual-e5-base",
    model_kwargs: dict[str, Any] = None,
    encode_kwargs: dict[str, Any] = {"normalize_embeddings": False},
) -> np.ndarray:
    from langchain_huggingface import HuggingFaceEmbeddings

    if model_kwargs is None:
        model_kwargs = {}
    if model_kwargs.get("device") is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs["device"] = device

    client = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


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
    if corpus_path.exists() and query_path.exists():
        print("キャッシュされた埋め込みを読み込みます...")
        with open(corpus_path, "rb") as f:
            corpus_embeddings = pickle.load(f)
        with open(query_path, "rb") as f:
            query_embeddings = pickle.load(f)
    else:
        print("埋め込みを計算します...")
        corpus_embeddings = get_openai_embeddings(
            full_corpus["text"], max_tokens_per_batch=4500000
        )
        query_embeddings = get_openai_embeddings(
            full_ds["query"], max_tokens_per_batch=4500000
        )
        print("埋め込みをキャッシュに保存します...")
        with open(corpus_path, "wb") as f:
            pickle.dump(corpus_embeddings, f)
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
