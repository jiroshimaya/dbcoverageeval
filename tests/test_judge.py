"""
judgeモジュールのテスト。
"""
import os
import pytest
import pandas as pd
import json

from dbcoverageeval.judge import Judge, judge


@pytest.fixture
def chunks_df_with_text():
    """テスト用のテキスト付きチャンクDataFrameを提供するフィクスチャ"""
    chunks = [
        {
            "chunk_id": "doc1_p1_c1",
            "parent_id": "doc1_p1",
            "doc_id": "doc1",
            "page_num": 1,
            "chunk_num": 1,
            "text": "これはテスト用のドキュメント1の1ページ目のチャンク1です。質問Aに関する情報が含まれています。"
        },
        {
            "chunk_id": "doc1_p1_c2",
            "parent_id": "doc1_p1",
            "doc_id": "doc1",
            "page_num": 1,
            "chunk_num": 2,
            "text": "これはテスト用のドキュメント1の1ページ目のチャンク2です。質問Bに関する部分的な情報が含まれています。"
        },
        {
            "chunk_id": "doc2_p1_c1",
            "parent_id": "doc2_p1",
            "doc_id": "doc2",
            "page_num": 1,
            "chunk_num": 1,
            "text": "これはテスト用のドキュメント2の1ページ目のチャンク1です。質問Cに関する情報は含まれていません。"
        }
    ]
    return pd.DataFrame(chunks)


@pytest.fixture
def parent_results():
    """テスト用の親ドキュメント検索結果を提供するフィクスチャ"""
    return [
        {
            "parent_id": "doc1_p1",
            "doc_id": "doc1",
            "page_num": 1,
            "avg_score": 0.85,
            "max_score": 0.9,
            "chunks": [
                {"chunk_id": "doc1_p1_c1", "score": 0.9, "rank": 1},
                {"chunk_id": "doc1_p1_c2", "score": 0.8, "rank": 2}
            ]
        },
        {
            "parent_id": "doc2_p1",
            "doc_id": "doc2",
            "page_num": 1,
            "avg_score": 0.7,
            "max_score": 0.7,
            "chunks": [
                {"chunk_id": "doc2_p1_c1", "score": 0.7, "rank": 3}
            ]
        }
    ]


@pytest.mark.usefixtures("mock_openai_chat_completions")
class TestJudge:
    """判定器のテスト"""

    def test_init(self, sample_config):
        """初期化のテスト"""
        judge_obj = Judge(sample_config)
        assert judge_obj.judge_model == sample_config["models"]["judge"]
        assert judge_obj.provider == sample_config["provider"]

    def test_prepare_excerpts(self, sample_config, chunks_df_with_text):
        """抜粋作成のテスト"""
        judge_obj = Judge(sample_config)
        parent_ids = ["doc1_p1", "doc2_p1"]
        excerpts = judge_obj._prepare_excerpts(chunks_df_with_text, parent_ids)
        
        # 結果の検証
        assert isinstance(excerpts, dict)
        assert len(excerpts) == 2
        assert "doc1_p1" in excerpts
        assert "doc2_p1" in excerpts
        
        # 抜粋の内容確認
        assert "質問A" in excerpts["doc1_p1"]
        assert "質問C" in excerpts["doc2_p1"]

    def test_judge_yes(self, sample_config, chunks_df_with_text, parent_results):
        """Yes判定のテスト"""
        judge_obj = Judge(sample_config)
        question = "質問A"
        result = judge_obj.judge(question, parent_results, chunks_df_with_text, max_parents=2)
        
        # 結果の検証
        assert isinstance(result, dict)
        assert "coverage" in result
        assert "reason" in result
        assert result["coverage"] == "Yes"

    def test_judge_partial(self, sample_config, chunks_df_with_text, parent_results):
        """Partial判定のテスト"""
        judge_obj = Judge(sample_config)
        question = "質問B"
        result = judge_obj.judge(question, parent_results, chunks_df_with_text, max_parents=2)
        
        # 結果の検証
        assert isinstance(result, dict)
        assert "coverage" in result
        assert "reason" in result
        assert result["coverage"] == "Partial"

    def test_judge_no(self, sample_config, chunks_df_with_text, parent_results):
        """No判定のテスト"""
        judge_obj = Judge(sample_config)
        question = "質問C"
        result = judge_obj.judge(question, parent_results, chunks_df_with_text, max_parents=2)
        
        # 結果の検証
        assert isinstance(result, dict)
        assert "coverage" in result
        assert "reason" in result
        assert result["coverage"] == "No"


@pytest.mark.usefixtures("mock_openai_chat_completions")
def test_judge_function(config_file, chunks_df_with_text, parent_results):
    """judge関数のテスト"""
    question = "質問A"
    result = judge(question, parent_results, chunks_df_with_text, config_file)
    
    # 結果の検証
    assert isinstance(result, dict)
    assert "coverage" in result
    assert "reason" in result
    assert result["coverage"] == "Yes"
