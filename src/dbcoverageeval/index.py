"""
インデックス構築モジュール。
Parquet形式のテキストチャンクからFAISSベクトルインデックスを構築する。
"""
import os
import numpy as np
import pandas as pd
import faiss
from typing import Dict, Any, List, Optional, Tuple
import pickle
from pathlib import Path
import yaml
import tiktoken

class IndexBuilder:
    """FAISSインデックスを構築するクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定情報（APIキー、モデル名など）
        """
        self.config = config
        self.embedding_model = config["models"]["embedding"]
        self.provider = config["provider"]
        
        # APIクライアントの初期化
        if self.provider == "openai":
            import openai
            openai.api_key = config["openai"]["api_key"]
            openai.api_base = config["openai"]["api_base"]
            self.client = openai
        elif self.provider == "azure_openai":
            import openai
            openai.api_key = config["azure_openai"]["api_key"]
            openai.api_base = config["azure_openai"]["api_base"]
            openai.api_type = "azure"
            openai.api_version = config["azure_openai"]["api_version"]
            self.client = openai
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config["gemini"]["api_key"])
            self.client = genai
        elif self.provider == "claude":
            import anthropic
            self.client = anthropic.Anthropic(api_key=config["claude"]["api_key"])
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        テキストのエンベディングを取得する
        
        Args:
            texts: エンベディング対象のテキストリスト
            
        Returns:
            テキストのエンベディング（N x D行列）
        """
        embeddings = []
        
        if self.provider == "openai" or self.provider == "azure_openai":
            # OpenAI API でエンベディングを取得
            batch_size = 100  # APIコール制限に配慮
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
        
        elif self.provider == "gemini":
            # Gemini API でエンベディングを取得
            for text in texts:
                response = self.client.embed_content(
                    model=self.embedding_model,
                    content=text
                )
                embeddings.append(response["embedding"])
        
        elif self.provider == "claude":
            # Claude API でエンベディングを取得
            for text in texts:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                embeddings.append(response.embedding)
        
        return np.array(embeddings, dtype=np.float32)

def build_index(parquet_path: str, out_index_dir: str, config_path: str = None) -> None:
    """
    Parquetファイルからベクトルインデックスを構築する
    
    Args:
        parquet_path: 入力Parquetファイルのパス
        out_index_dir: 出力インデックスディレクトリのパス
        config_path: 設定ファイルのパス（デフォルト: None）
    """
    # 設定ファイルの読み込み
    if config_path is None:
        config_path = "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parquetファイルの読み込み
    df = pd.read_parquet(parquet_path)
    texts = df["text"].tolist()
    
    print(f"Building index for {len(texts)} text chunks...")
    
    # インデクシング用ディレクトリの作成
    out_index_dir = Path(out_index_dir)
    out_index_dir.mkdir(parents=True, exist_ok=True)
    
    # エンベディングの取得
    index_builder = IndexBuilder(config)
    embeddings = index_builder.get_embeddings(texts)
    
    # FAISSインデックスの構築
    dimension = embeddings.shape[1]
    
    # IVFFlatインデックスの構築（効率的な検索のため）
    quantizer = faiss.IndexFlatIP(dimension)  # 内積類似度ベース
    nlist = min(4096, len(texts) // 10)  # クラスタ数を調整
    nlist = max(nlist, 1)  # 少なくとも1つのクラスタ
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # 正規化（内積→コサイン類似度）
    faiss.normalize_L2(embeddings)
    
    # 訓練とデータ追加
    if not index.is_trained:
        index.train(embeddings)
    index.add(embeddings)
    
    # インデックスの保存
    faiss.write_index(index, str(out_index_dir / "vector.index"))
    
    # メタデータの保存
    meta_df = df[["chunk_id", "parent_id", "doc_id", "page_num", "chunk_num"]]
    meta_df.to_parquet(str(out_index_dir / "meta.parquet"))
    
    # 設定情報の保存
    with open(str(out_index_dir / "index_config.pkl"), 'wb') as f:
        pickle.dump({
            "dimension": dimension,
            "num_chunks": len(texts),
            "embedding_model": config["models"]["embedding"]
        }, f)
    
    print(f"Index built successfully. Files saved to {out_index_dir}")
