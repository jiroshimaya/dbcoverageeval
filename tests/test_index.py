"""
indexモジュールのテスト。
"""
import os
import pytest
import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path

from dbcoverageeval.index import IndexBuilder, build_index


@pytest.mark.usefixtures("mock_openai_embeddings")
class TestIndexBuilder:
    """インデックスビルダーのテスト"""

    def test_init(self, sample_config):
        """初期化のテスト"""
        builder = IndexBuilder(sample_config)
        assert builder.embedding_model == sample_config["models"]["embedding"]
        assert builder.provider == sample_config["provider"]

    def test_get_embeddings(self, sample_config):
        """エンベディング取得のテスト"""
        builder = IndexBuilder(sample_config)
        texts = ["This is a test document.", "Another test document."]
        embeddings = builder.get_embeddings(texts)
        
        # 形状の検証
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 128)  # モックでは128次元
        
        # 値の検証（正規化されているため、ノルムは1に近いはず）
        for i in range(embeddings.shape[0]):
            norm = np.linalg.norm(embeddings[i])
            assert 0.99 <= norm <= 1.01


@pytest.mark.usefixtures("mock_openai_embeddings")
def test_build_index(temp_dir, chunks_parquet, config_file):
    """インデックス構築機能のテスト"""
    # インデックス出力ディレクトリ
    out_index_dir = os.path.join(temp_dir, "index")
    
    # インデックス構築
    build_index(chunks_parquet, out_index_dir, config_file)
    
    # 必要なファイルが作成されたか確認
    assert os.path.exists(os.path.join(out_index_dir, "vector.index"))
    assert os.path.exists(os.path.join(out_index_dir, "meta.parquet"))
    assert os.path.exists(os.path.join(out_index_dir, "index_config.pkl"))
    
    # インデックス設定の確認
    with open(os.path.join(out_index_dir, "index_config.pkl"), 'rb') as f:
        index_config = pickle.load(f)
    
    assert "dimension" in index_config
    assert "num_chunks" in index_config
    assert "embedding_model" in index_config
    
    # メタデータの確認
    meta_df = pd.read_parquet(os.path.join(out_index_dir, "meta.parquet"))
    assert "chunk_id" in meta_df.columns
    assert "parent_id" in meta_df.columns
    
    # インデックスの読み込みテスト
    index = faiss.read_index(os.path.join(out_index_dir, "vector.index"))
    assert index.ntotal > 0  # インデックスにベクトルが含まれていることを確認
