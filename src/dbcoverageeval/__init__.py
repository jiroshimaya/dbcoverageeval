"""
質問カバー率測定パイプライン。
ドキュメントセットと質問リストから質問がドキュメントでカバーされるか判定する。
"""
from typing import List, Dict, Any

__version__ = "0.1.0"

from .ingest import ingest
from .index import build_index
from .querygen import expand_query
from .search import retrieve
from .judge import judge
from .report import save
