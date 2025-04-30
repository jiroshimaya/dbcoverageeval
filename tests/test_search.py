"""
searchモジュールのテスト。
"""
import os
import pytest
import faiss
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from dbcoverageeval.search import Searcher, retrieve


@pytest.fixture
def index_dir(temp_dir, mock_openai_embeddings):
    """テスト用のインデックスディレクトリを作成するフィクスチャ"""
    # インデックスディレクトリ
    index_dir = os.path.join(temp_dir, "index")
    os.makedirs(index_dir, exist_ok=True)
    
    # サンプルのチャンクデータ
    chunks = [
        {
            "chunk_id": "doc1_p1_c1",
            "parent_id": "doc1_p1",
            "doc_id": "doc1",
            "page_num": 1,
            "chunk_num": 1
        },
        {
            "chunk_id": "doc1_p1_c2",
            "parent_id": "doc1_p1",
            "doc_id": "doc1",
            "page_num": 1,
            "chunk_num": 2
        },
        {
            "chunk_id": "doc2_p1_c1",
            "parent_id": "doc2_p1",
            "doc_id": "doc2",
            "page_num": 1,
            "chunk_num": 1
        }
    ]
    
    # メタデータの保存
    meta_df = pd.DataFrame(chunks)
    meta_df.to_parquet(os.path.join(index_dir, "meta.parquet"))
    
    # ダミーの埋め込みベクトル
    dimension = 128
    embeddings = np.random.rand(len(chunks), dimension).astype(np.float32)
    faiss.normalize_L2(embeddings)
    
    # FAISSインデックスの作成
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "vector.index"))
    
    # インデックス設定の保存
    with open(os.path.join(index_dir, "index_config.pkl"), 'wb') as f:
        pickle.dump({
            "dimension": dimension,
            "num_chunks": len(chunks),
            "embedding_model": "text-embedding-3-small"
        }, f)
    
    return index_dir


@pytest.mark.usefixtures("mock_openai_embeddings")
class TestSearcher:
    """検索器のテスト"""

    def test_init(self, sample_config, index_dir):
        """初期化のテスト"""
        searcher = Searcher(index_dir, sample_config)
        assert searcher.provider == sample_config["provider"]
        assert searcher.embedding_model == "text-embedding-3-small"

    def test_get_embedding(self, sample_config, index_dir):
        """エンベディング取得のテスト"""
        searcher = Searcher(index_dir, sample_config)
        query = "テスト用のクエリ"
        embedding = searcher.get_embedding(query)
        
        # 形状の検証
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 128)  # モックでは128次元
        
        # 正規化の検証
        norm = np.linalg.norm(embedding[0])
        assert 0.99 <= norm <= 1.01

    def test_search_chunks(self, sample_config, index_dir):
        """チャンク検索のテスト"""
        searcher = Searcher(index_dir, sample_config)
        query = "テスト用のクエリ"
        results = searcher.search_chunks(query, k=2)
        
        # 形式の検証
        assert isinstance(results, list)
        assert len(results) <= 2  # 最大k件
        
        if results:  # 結果がある場合
            # 結果の各要素の検証
            for result in results:
                assert "chunk_id" in result
                assert "score" in result
                assert "rank" in result
                assert isinstance(result["score"], float)
                assert isinstance(result["rank"], int)

    def test_group_chunks_by_parent(self, sample_config, index_dir):
        """親ドキュメントごとのグループ化テスト"""
        searcher = Searcher(index_dir, sample_config)
        
        # ダミーのチャンク検索結果
        chunks = [
            {"parent_id": "doc1_p1", "doc_id": "doc1", "page_num": 1, "score": 0.9, "rank": 1},
            {"parent_id": "doc1_p1", "doc_id": "doc1", "page_num": 1, "score": 0.8, "rank": 2},
            {"parent_id": "doc2_p1", "doc_id": "doc2", "page_num": 1, "score": 0.7, "rank": 3}
        ]
        
        results = searcher._group_chunks_by_parent(chunks, top_k_parent=2)
        
        # 結果の検証
        assert isinstance(results, list)
        assert len(results) <= 2  # 最大top_k_parent件
        
        # 各親ドキュメントの検証
        for parent in results:
            assert "parent_id" in parent
            assert "max_score" in parent
            assert "avg_score" in parent
            assert "chunks" in parent
            assert isinstance(parent["chunks"], list)

    def test_retrieve(self, sample_config, index_dir):
        """複数クエリ検索のテスト"""
        searcher = Searcher(index_dir, sample_config)
        queries = ["クエリ1", "クエリ2"]
        results = searcher.retrieve(queries, k=2, top_k_parent=2)
        
        # 結果の検証
        assert isinstance(results, list)
        assert len(results) <= 2  # 最大top_k_parent件
        
        # 結果の形式の検証
        if results:
            parent = results[0]
            assert "parent_id" in parent
            assert "max_score" in parent
            assert "chunks" in parent


@pytest.mark.usefixtures("mock_openai_embeddings")
def test_retrieve_function(config_file, index_dir):
    """retrieve関数のテスト"""
    queries = ["テスト用のクエリ"]
    results = retrieve(queries, index_dir, config_file, k=2)
    
    # 結果の検証
    assert isinstance(results, list)
    
    # 結果の形式の検証（結果がある場合）
    if results:
        parent = results[0]
        assert "parent_id" in parent
        assert "max_score" in parent
        assert "chunks" in parent
