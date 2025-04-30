"""
レポート生成モジュール。
評価結果を集計してJSON形式で出力する。
"""
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

def _prepare_retrieved_docs(parent_results: List[Dict[str, Any]], chunks_df: pd.DataFrame) -> Dict[str, str]:
    """
    検索結果のドキュメント抜粋を作成する
    
    Args:
        parent_results: 親ドキュメント情報のリスト
        chunks_df: チャンクのDataFrame
        
    Returns:
        親ドキュメントIDをキー、抜粋を値とする辞書
    """
    docs = {}
    for parent in parent_results:
        parent_id = parent["parent_id"]
        
        # 親に所属するチャンクを取得
        parent_chunks = chunks_df[chunks_df["parent_id"] == parent_id]
        
        if len(parent_chunks) == 0:
            continue
        
        # テキストを連結（最大1000文字）
        text = " ".join([chunk["text"] for chunk in parent_chunks.iloc[:3].to_dict('records')])
        if len(text) > 1000:
            text = text[:997] + "..."
        
        docs[parent_id] = text
    
    return docs

def _calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    評価結果から指標を計算する
    
    Args:
        results: 評価結果のリスト
        
    Returns:
        指標の辞書
    """
    # カバレッジのカウント
    coverage_counts = {"Yes": 0, "Partial": 0, "No": 0}
    
    for result in results:
        coverage = result["coverage"]
        coverage_counts[coverage] += 1
    
    # 合計数
    total = sum(coverage_counts.values())
    
    # 比率を計算
    metrics = {
        "yes": coverage_counts["Yes"],
        "partial": coverage_counts["Partial"],
        "no": coverage_counts["No"],
        "coverage_rate_yes": round(coverage_counts["Yes"] / total, 3) if total > 0 else 0,
        "coverage_rate_yes_or_partial": round((coverage_counts["Yes"] + coverage_counts["Partial"]) / total, 3) if total > 0 else 0
    }
    
    return metrics

def save(json_path: str, 
         questions: List[Dict[str, Any]], 
         parent_results_map: Dict[str, List[Dict[str, Any]]], 
         chunks_df: pd.DataFrame, 
         judgments: Dict[str, Dict[str, Any]]) -> None:
    """
    評価結果をJSONファイルに保存する
    
    Args:
        json_path: 出力するJSONファイルのパス
        questions: 質問のリスト（id, questionを含む）
        parent_results_map: 質問IDをキー、親ドキュメント情報のリストを値とする辞書
        chunks_df: チャンクのDataFrame
        judgments: 質問IDをキー、判定結果を値とする辞書
    """
    # 取得したドキュメントの抜粋
    retrieved_docs = {}
    for q_id, parent_results in parent_results_map.items():
        docs = _prepare_retrieved_docs(parent_results, chunks_df)
        retrieved_docs.update(docs)
    
    # 質問ごとの評価結果
    questions_with_results = []
    for q in questions:
        q_id = q["id"]
        
        # 関連するドキュメントIDを取得
        doc_ids = []
        if q_id in parent_results_map:
            doc_ids = [parent["parent_id"] for parent in parent_results_map[q_id][:3]]
        
        # 判定結果を取得
        coverage = "No"
        reason = "判定できませんでした"
        if q_id in judgments:
            coverage = judgments[q_id]["coverage"]
            reason = judgments[q_id]["reason"]
        
        # 質問と評価結果を結合
        questions_with_results.append({
            "id": q_id,
            "question": q["question"],
            "doc_ids": doc_ids,
            "coverage": coverage,
            "reason": reason
        })
    
    # 指標を計算
    metrics = _calculate_metrics(judgments.values())
    
    # 結果を構築
    results = {
        "retrieved_docs": retrieved_docs,
        "questions": questions_with_results,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # JSONとして保存
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {json_path}")
    print(f"Summary: Yes={metrics['yes']}, Partial={metrics['partial']}, No={metrics['no']}")
    print(f"Coverage Rate (Yes): {metrics['coverage_rate_yes']:.1%}")
    print(f"Coverage Rate (Yes or Partial): {metrics['coverage_rate_yes_or_partial']:.1%}")
