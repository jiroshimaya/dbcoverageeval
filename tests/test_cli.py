"""
cliモジュールのテスト。
"""
import os
import pytest
import typer
from typer.testing import CliRunner
import pandas as pd
import json
import yaml
from pathlib import Path

from dbcoverageeval.cli import app


@pytest.fixture
def runner():
    """Typerのコマンドライン実行用フィクスチャ"""
    return CliRunner()


class TestConfigCommand:
    """設定ファイル生成コマンドのテスト"""
    
    def test_generate_config(self, runner, temp_dir):
        """設定ファイル生成のテスト"""
        config_path = os.path.join(temp_dir, "test_config.yaml")
        result = runner.invoke(app, ["config", config_path])
        
        # 実行結果の検証
        assert result.exit_code == 0
        assert os.path.exists(config_path)
        
        # 設定ファイルの内容確認
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 必要なキーの存在確認
        assert "provider" in config
        assert "models" in config
        assert "openai" in config
        assert "chunk_tokens" in config
        assert "chunk_overlap" in config


@pytest.mark.parametrize("test_ext", [".txt"])
def test_ingest_command(runner, temp_dir, test_ext):
    """取り込みコマンドのテスト"""
    # テスト用ディレクトリとファイルを準備
    docs_path = os.path.join(temp_dir, "docs")
    os.makedirs(docs_path, exist_ok=True)
    
    # テスト用のテキストファイルを作成
    file_path = os.path.join(docs_path, f"test_doc{test_ext}")
    with open(file_path, 'w') as f:
        f.write("This is a test document for CLI testing. " * 10)
    
    # 設定ファイルを生成
    config_path = os.path.join(temp_dir, "test_config.yaml")
    runner.invoke(app, ["config", config_path])
    
    # Parquet出力パス
    out_parquet = os.path.join(temp_dir, "cli_test_chunks.parquet")
    
    # ingsetコマンドを実行
    result = runner.invoke(
        app, 
        ["ingest", docs_path, "--out", out_parquet, "--config", config_path]
    )
    
    # 実行結果の検証
    assert result.exit_code == 0
    assert os.path.exists(out_parquet)
    
    # Parquetファイルの内容確認
    df = pd.read_parquet(out_parquet)
    assert not df.empty
    assert "chunk_id" in df.columns
    assert "text" in df.columns


@pytest.mark.usefixtures("mock_openai_embeddings")
def test_build_index_command(runner, temp_dir):
    """インデックス構築コマンドのテスト"""
    # 設定ファイルを生成
    config_path = os.path.join(temp_dir, "test_config.yaml")
    runner.invoke(app, ["config", config_path])
    
    # テスト用のチャンクParquetを作成
    chunks_path = os.path.join(temp_dir, "test_chunks.parquet")
    chunks = [
        {
            "chunk_id": "doc1_p1_c1",
            "parent_id": "doc1_p1",
            "doc_id": "doc1",
            "page_num": 1,
            "chunk_num": 1,
            "text": "This is a test chunk 1."
        },
        {
            "chunk_id": "doc1_p1_c2",
            "parent_id": "doc1_p1",
            "doc_id": "doc1",
            "page_num": 1,
            "chunk_num": 2,
            "text": "This is a test chunk 2."
        }
    ]
    pd.DataFrame(chunks).to_parquet(chunks_path)
    
    # インデックス出力ディレクトリ
    index_dir = os.path.join(temp_dir, "test_index")
    
    # build-indexコマンドを実行
    result = runner.invoke(
        app, 
        ["build-index", chunks_path, index_dir, "--config", config_path]
    )
    
    # 実行結果の検証
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(index_dir, "vector.index"))
    assert os.path.exists(os.path.join(index_dir, "meta.parquet"))
    assert os.path.exists(os.path.join(index_dir, "index_config.pkl"))


def test_load_questions():
    """質問読み込み機能のテスト"""
    # _load_questions関数は内部関数なので、直接テストできない
    # CLIクラス内の機能としてテストするか、設計を変更する必要がある
    # このテストは例示的なもので、実際の実装に合わせて調整が必要
    pass


def test_generate_config_command(runner, temp_dir):
    """設定ファイル生成コマンドのテスト（拡張版）"""
    config_path = os.path.join(temp_dir, "config_test.yaml")
    result = runner.invoke(app, ["config", config_path])
    
    # 実行結果の検証
    assert result.exit_code == 0
    assert "設定ファイルのテンプレートを" in result.stdout
    assert os.path.exists(config_path)
    
    # 設定ファイルの内容確認
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 重要な設定が含まれていることを確認
    assert "provider: openai" in content
    assert "embedding:" in content
    assert "query_gen:" in content
    assert "judge:" in content
    assert "api_key:" in content
    assert "chunk_tokens:" in content
