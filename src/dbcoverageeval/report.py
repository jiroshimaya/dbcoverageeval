"""
レポート生成モジュール。
評価結果を集計してJSON形式で出力する。
"""
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel

from dbcoverageeval.judge import DetailedJudgeResult


class DetailedResult(BaseModel):
    """詳細な評価結果"""
    question: str
    reason: str
    judge: str
    search_results: list[str]

class Report(BaseModel):
    """評価結果"""
    summary: Dict[str, Any]
    details: list[DetailedResult]
    documents: Dict[str, Any]
    
    def save(self, output_path: str):
        with open(output_path, "w") as f:
            json.dump(self.model_dump(), f, indent = 2, ensure_ascii=False)

def calculate_metrics(judge_results: list[DetailedJudgeResult]) -> Report:
    """
    評価結果を集計してJSON形式で出力する
    """
    
    summary = {
        "count": {},
        "ratio": {}
    }
    documents = {}
    details = []
    
    for judge_result in judge_results:
        summary["count"][judge_result.judge] = summary["count"].get(judge_result.judge, 0) + 1
        summary["count"]["total"] = summary["count"].get("total", 0) + 1
        
        documents.update({doc.metadata["id"]: doc.page_content for doc in judge_result.search_results})
        
        detail_result = DetailedResult(question=judge_result.question, reason=judge_result.reason, judge=judge_result.judge, search_results=[doc.metadata["id"] for doc in judge_result.search_results])
        details.append(detail_result)

    summary["ratio"] = {k: v / summary["count"]["total"] for k, v in summary["count"].items()}
    return Report(summary=summary, details=details, documents=documents)

