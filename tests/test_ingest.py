"""
ingestモジュールのテスト。
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document

from dbcoverageeval.ingest import ingest_documents, load_pdf_files


def test_load_pdf_files():
    docs_path = Path(__file__).parent / "data"
    documents = load_pdf_files(docs_path)
    
    assert len(documents) == 2
    assert documents[0].page_content == "タイトル\nサブタイトル"
    assert documents[1].page_content == "ページ１\nページ１コンテンツ"

