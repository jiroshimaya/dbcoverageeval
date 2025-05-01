"""
質問カバー率測定パイプライン。
ドキュメントセットと質問リストから質問がドキュメントでカバーされるか判定する。
"""
from typing import Any, Dict, List

__version__ = "0.1.0"

from .ingest import ingest
from .judge import judge_search_result
