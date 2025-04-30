"""
ingestモジュールのテスト。
"""
import os
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import tiktoken

from dbcoverageeval.ingest import (
    Extractor, TextExtractor, chunk_text, ingest, get_extractor
)


class TestChunkText:
    """テキストチャンク化機能のテスト"""

    def test_chunk_text_with_default_params(self):
        """デフォルトパラメータでのテキストチャンク化をテスト"""
        text = "a" * 5000  # tiktoken でエンコードすると1000トークン以上になる十分な長さのテキスト
        chunks = chunk_text(text)
        
        # チャンクが作成されたことを確認
        assert len(chunks) > 1
        
        # 各チャンクが最大トークン数を超えていないことを確認
        tokenizer = tiktoken.get_encoding("cl100k_base")
        for chunk in chunks:
            assert len(tokenizer.encode(chunk)) <= 1000

    def test_chunk_text_with_custom_params(self):
        """カスタムパラメータでのテキストチャンク化をテスト"""
        text = "a" * 2000  # 短いテキスト
        chunks = chunk_text(text, chunk_tokens=200, overlap_tokens=50)
        
        # チャンクが作成されたことを確認
        assert len(chunks) > 1
        
        # 各チャンクが最大トークン数を超えていないことを確認
        tokenizer = tiktoken.get_encoding("cl100k_base")
        for chunk in chunks:
            assert len(tokenizer.encode(chunk)) <= 200

    def test_chunk_text_short_input(self):
        """短いテキスト入力でのチャンク化をテスト"""
        text = "This is a short text."
        chunks = chunk_text(text)
        
        # 短いテキストは1つのチャンクになることを確認
        assert len(chunks) == 1
        assert chunks[0] == text


class TestExtractors:
    """テキスト抽出器のテスト"""

    def test_text_extractor(self):
        """TextExtractorのテスト"""
        # テスト用の一時ファイルを作成
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
            f.write("This is a test document.")
            file_path = f.name
        
        try:
            # TextExtractorでテキストを抽出
            extractor = TextExtractor()
            results = extractor.extract(file_path)
            
            # 結果を検証
            assert len(results) == 1
            assert results[0]["text"] == "This is a test document."
            assert results[0]["page_num"] == 1
            assert results[0]["doc_id"] == os.path.basename(file_path).split('.')[0]
        finally:
            # 一時ファイルを削除
            os.unlink(file_path)

    def test_get_extractor(self):
        """get_extractor関数のテスト"""
        # 各種ファイル形式でのExtractor取得をテスト
        assert isinstance(get_extractor("test.txt"), TextExtractor)
        assert isinstance(get_extractor("test.md"), TextExtractor)
        assert get_extractor("test.unknown") is None


@pytest.mark.parametrize("file_ext", [".txt", ".md"])
def test_ingest_with_text_files(temp_dir, file_ext):
    """テキストファイルの取り込みテスト"""
    # テスト用ディレクトリとファイルを準備
    docs_path = os.path.join(temp_dir, "docs")
    os.makedirs(docs_path, exist_ok=True)
    
    # テスト用のテキストファイルを作成
    file_path = os.path.join(docs_path, f"test_doc{file_ext}")
    with open(file_path, 'w') as f:
        f.write("This is a test document with multiple sentences. " * 10)
    
    # Parquet出力パス
    out_parquet = os.path.join(temp_dir, "chunks.parquet")
    
    # ingest関数を実行
    ingest(docs_path, out_parquet, chunk_tokens=100, chunk_overlap=20)
    
    # 結果を検証
    assert os.path.exists(out_parquet)
    
    # Parquetファイルを読み込み
    df = pd.read_parquet(out_parquet)
    
    # 基本的な検証
    assert not df.empty
    assert "chunk_id" in df.columns
    assert "parent_id" in df.columns
    assert "doc_id" in df.columns
    assert "page_num" in df.columns
    assert "chunk_num" in df.columns
    assert "text" in df.columns
    
    # 内容の検証
    assert df["doc_id"].iloc[0] == "test_doc"
    assert df["page_num"].iloc[0] == 1
    assert "This is a test document" in df["text"].iloc[0]
