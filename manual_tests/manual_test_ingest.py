"""
ingestモジュールのテスト。
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document

from dbcoverageeval.ingest import ingest, ingest_documents


def test_ingest_documents():
    documents = [
        Document(page_content="タイトル\nサブタイトル"),
        Document(page_content="ページ１\nページ１コンテンツ"),
    ]
    db = ingest_documents(documents)
    assert db is not None
    assert db.similarity_search("タイトル")[0].page_content == documents[0].page_content
    assert db.similarity_search("ページ１")[0].page_content == documents[1].page_content

def test_ingest():
    docs_path = Path(__file__).parent / "data"
    db = ingest(docs_path)
    assert db is not None
    assert db.similarity_search("タイトル")[0].page_content == "タイトル\nサブタイトル"
    assert db.similarity_search("ページ１")[0].page_content == "ページ１\nページ１コンテンツ"