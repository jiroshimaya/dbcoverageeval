"""
検索モジュール。
クエリをベクトル検索し、関連するドキュメントを取得する。
"""
import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import yaml
from collections import defaultdict, Counter

class Searcher:
    """FAISSインデックスを用いて検索を行うクラス"""
    
    def __init__(self, 
                 index_dir: str, 
                 config: Dict[str, Any]):
        """
        初期化
        
        Args:
            index_dir: インデックスディレクトリのパス
            config: 設定情報
        """
        self.config = config
        self.index_dir = Path(index_dir)
        
        # FAISSインデックスの読み込み
        self.index = faiss.read_index(str(self.index_dir / "vector.index"))
        
        # メタデータの読み込み
        self.meta_df = pd.read_parquet(str(self.index_dir / "meta.parquet"))
        self.chunk_id_to_idx = {chunk_id: i for i, chunk_id in enumerate(self.meta_df["chunk_id"])}
        
        # インデックス設定の読み込み
        with open(str(self.index_dir / "index_config.pkl"), 'rb') as f:
            self.index_config = pickle.load(f)
        
        # 埋め込みモデルの初期化
        self.embedding_model = self.index_config["embedding_model"]
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
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        テキストのエンベディングを取得する
        
        Args:
            text: エンベディング対象のテキスト
            
        Returns:
            テキストのエンベディング（1 x D行列）
        """
        if self.provider == "openai" or self.provider == "azure_openai":
            # OpenAI API でエンベディングを取得
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )
            embedding = response.data[0].embedding
        
        elif self.provider == "gemini":
            # Gemini API でエンベディングを取得
            response = self.client.embed_content(
                model=self.embedding_model,
                content=text
            )
            embedding = response["embedding"]
        
        elif self.provider == "claude":
            # Claude API でエンベディングを取得
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.embedding
        
        # 正規化（コサイン類似度のため）
        embedding_np = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(embedding_np)
        
        return embedding_np
    
    def search_chunks(self, query: str, k: int = 25) -> List[Dict[str, Any]]:
        """
        クエリに関連するチャンクを検索する
        
        Args:
            query: 検索クエリ
            k: 取得するチャンク数
            
        Returns:
            関連するチャンク情報のリスト
        """
        # クエリをベクトル化
        query_vector = self.get_embedding(query)
        
        # ベクトル検索
        self.index.nprobe = 20  # 探索クラスタ数
        scores, indices = self.index.search(query_vector, k)
        
        # 検索結果を構築
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISSの無効インデックス
                continue
                
            # メタデータの取得
            chunk_info = self.meta_df.iloc[idx].to_dict()
            chunk_info["score"] = float(score)
            chunk_info["rank"] = i + 1
            
            results.append(chunk_info)
        
        return results
    
    def _group_chunks_by_parent(self, chunks: List[Dict[str, Any]], top_k_parent: int = 100) -> List[Dict[str, Any]]:
        """
        チャンクを親ドキュメントごとにグループ化して上位を取得
        
        Args:
            chunks: チャンク情報のリスト
            top_k_parent: 取得する親ドキュメント数
            
        Returns:
            親ドキュメント情報のリスト（スコア降順）
        """
        # 親ごとにチャンクをグループ化
        parent_groups = defaultdict(list)
        for chunk in chunks:
            parent_id = chunk["parent_id"]
            parent_groups[parent_id].append(chunk)
        
        # 親ごとのスコアを計算（平均スコア）
        parent_scores = []
        for parent_id, parent_chunks in parent_groups.items():
            avg_score = sum(chunk["score"] for chunk in parent_chunks) / len(parent_chunks)
            max_score = max(chunk["score"] for chunk in parent_chunks)
            best_chunk = max(parent_chunks, key=lambda x: x["score"])
            
            parent_scores.append({
                "parent_id": parent_id,
                "doc_id": best_chunk["doc_id"],
                "page_num": best_chunk["page_num"],
                "avg_score": avg_score,
                "max_score": max_score,
                "chunks": parent_chunks
            })
        
        # スコア降順にソート
        parent_scores.sort(key=lambda x: x["max_score"], reverse=True)
        
        # 上位k件を返す
        return parent_scores[:top_k_parent]

    def retrieve(self, queries: List[str], k: int = 25, top_k_parent: int = 100) -> List[Dict[str, Any]]:
        """
        複数クエリに対して検索を実行し、関連する親ドキュメントを取得
        
        Args:
            queries: 検索クエリのリスト
            k: 各クエリで取得するチャンク数
            top_k_parent: 取得する親ドキュメント数
            
        Returns:
            親ドキュメント情報のリスト（スコア降順）
        """
        all_chunks = []
        
        # 各クエリで検索を実行
        for query in queries:
            chunks = self.search_chunks(query, k=k)
            all_chunks.extend(chunks)
        
        # 親ドキュメントごとにグループ化
        parent_results = self._group_chunks_by_parent(all_chunks, top_k_parent=top_k_parent)
        
        return parent_results


def retrieve(queries: List[str], index_dir: str, config_path: str = None, k: int = 25) -> List[Dict[str, Any]]:
    """
    クエリに対して検索を実行し、関連するドキュメントを取得する
    
    Args:
        queries: 検索クエリのリスト
        index_dir: インデックスディレクトリのパス
        config_path: 設定ファイルのパス（デフォルト: None）
        k: 各クエリで取得するチャンク数
        
    Returns:
        親ドキュメント情報のリスト（スコア降順）
    """
    # 設定ファイルの読み込み
    if config_path is None:
        config_path = "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    top_k_parent = config.get("top_k_parent", 100)
    
    # 検索器の初期化
    searcher = Searcher(index_dir, config)
    
    # 検索の実行
    parent_results = searcher.retrieve(queries, k=k, top_k_parent=top_k_parent)
    
    return parent_results
