"""
reportモジュールのテスト。
"""
import os
import pytest
import pandas as pd
import json
from pathlib import Path

from dbcoverageeval.report import _prepare_retrieved_docs, _calculate_metrics, save


@pytest.fixture
def judgments():
    """テスト用の判定結果を提供するフィクスチャ"""
    return {
        "q_001": {"coverage": "Yes", "reason": "ドキュメントに十分な情報があります"},
        "q_002": {"coverage": "Partial", "reason": "ドキュメントに部分的な情報があります"},
        "q_003": {"coverage": "No", "reason": "ドキュメントに情報がありません"}
    }


@pytest.fixture
def parent_results_map(parent_results):
    """テスト用の親ドキュメント検索結果マップを提供するフィクスチャ"""
    return {
        "q_001": parent_results[:1],  # 質問1には最初の親ドキュメントだけ
        "q_002": parent_results,      # 質問2には両方の親ドキュメント
        "q_003": parent_results[1:2]  # 質問3には2番目の親ドキュメントだけ
    }


class TestPrepareRetrievedDocs:
    """ドキュメント抜粋作成機能のテスト"""

    def test_prepare_retrieved_docs(self, parent_results, chunks_df_with_text):
        """ドキュメント抜粋作成のテスト"""
        docs = _prepare_retrieved_docs(parent_results, chunks_df_with_text)
        
        # 結果の検証
        assert isinstance(docs, dict)
        assert len(docs) == 2
        assert "doc1_p1" in docs
        assert "doc2_p1" in docs
        
        # 抜粋の内容確認
        assert "質問A" in docs["doc1_p1"]
        assert "質問C" in docs["doc2_p1"]
        assert len(docs["doc1_p1"]) <= 1000  # 最大長の確認
        assert len(docs["doc2_p1"]) <= 1000


class TestCalculateMetrics:
    """指標計算機能のテスト"""

    def test_calculate_metrics(self, judgments):
        """指標計算のテスト"""
        metrics = _calculate_metrics(judgments.values())
        
        # 結果の検証
        assert isinstance(metrics, dict)
        assert "yes" in metrics
        assert "partial" in metrics
        assert "no" in metrics
        assert "coverage_rate_yes" in metrics
        assert "coverage_rate_yes_or_partial" in metrics
        
        # 値の検証
        assert metrics["yes"] == 1
        assert metrics["partial"] == 1
        assert metrics["no"] == 1
        assert metrics["coverage_rate_yes"] == 1/3
        assert metrics["coverage_rate_yes_or_partial"] == 2/3


def test_save(temp_dir, sample_questions, parent_results_map, chunks_df_with_text, judgments):
    """結果保存機能のテスト"""
    # 出力JSONパス
    out_json = os.path.join(temp_dir, "results.json")
    
    # 保存関数の実行
    save(out_json, sample_questions, parent_results_map, chunks_df_with_text, judgments)
    
    # ファイルが作成されたか確認
    assert os.path.exists(out_json)
    
    # JSONファイルを読み込み
    with open(out_json, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    # 結果の構造の検証
    assert "retrieved_docs" in result
    assert "questions" in result
    assert "metrics" in result
    assert "timestamp" in result
    
    # 質問の検証
    assert len(result["questions"]) == 3
    for q in result["questions"]:
        assert "id" in q
        assert "question" in q
        assert "doc_ids" in q
        assert "coverage" in q
        assert "reason" in q
    
    # 指標の検証
    metrics = result["metrics"]
    assert metrics["yes"] == 1
    assert metrics["partial"] == 1
    assert metrics["no"] == 1
    assert metrics["coverage_rate_yes"] == 1/3
    assert metrics["coverage_rate_yes_or_partial"] == 2/3
