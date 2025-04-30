"""
querygenモジュールのテスト。
"""
import os
import pytest
import json

from dbcoverageeval.querygen import QueryGenerator, expand_query


@pytest.mark.usefixtures("mock_openai_chat_completions")
class TestQueryGenerator:
    """クエリ生成器のテスト"""

    def test_init(self, sample_config):
        """初期化のテスト"""
        generator = QueryGenerator(sample_config)
        assert generator.query_gen_model == sample_config["models"]["query_gen"]
        assert generator.provider == sample_config["provider"]

    def test_expand_query(self, sample_config):
        """クエリ拡張のテスト"""
        generator = QueryGenerator(sample_config)
        question = "質問Aに関する情報を教えてください"
        queries = generator.expand_query(question)
        
        # 形式の検証
        assert isinstance(queries, list)
        assert len(queries) == 4
        
        # 各クエリが文字列であることを確認
        for query in queries:
            assert isinstance(query, str)


@pytest.mark.usefixtures("mock_openai_chat_completions")
def test_expand_query_function(config_file):
    """expand_query関数のテスト"""
    question = "質問Bに関する情報を教えてください"
    queries = expand_query(question, config_file)
    
    # 形式の検証
    assert isinstance(queries, list)
    assert len(queries) == 4
    
    # 各クエリが文字列であることを確認
    for query in queries:
        assert isinstance(query, str)
