"""
ingestモジュールのテスト。
"""
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document


def create_db(persist_dir: Optional[str] = None):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    if persist_dir is None:
        db = Chroma(embedding_function=OpenAIEmbeddings())
    else:
        db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=persist_dir)
    return db

def test_load_or_create_db():
    from dbcoverageeval.ingest import load_or_create_db
    with tempfile.TemporaryDirectory() as temp_dir:
        db = load_or_create_db(temp_dir)
        assert db._collection.count() == 0
        
        db.add_documents([Document(page_content="テスト")])
        assert db._collection.count() == 1
                
        db = load_or_create_db(temp_dir)
        assert db._collection.count() == 1

def test_add_documents():
    from dbcoverageeval.ingest import add_documents
    documents = [
        Document(page_content="テスト"),
    ]
    db = create_db()
    db.add_documents(documents)
    assert db._collection.count() == 1
    
    add_documents(db, documents)
    assert db._collection.count() == 1
    
    add_documents(db, [Document(page_content="テスト2")])
    assert db._collection.count() == 2

def test_reset_db():
    from dbcoverageeval.ingest import reset_db
    db = create_db()
    db.add_documents([Document(page_content="テスト")])
    assert db._collection.count() == 1
    reset_db(db)
    assert db._collection.count() == 0

def test_ingest():
    from dbcoverageeval.ingest import ingest
    docs_path = Path(__file__).parent / "data"
    db = ingest(docs_path)
    assert db is not None
    assert db.similarity_search("タイトル")[0].page_content == "タイトル\nサブタイトル"
    assert db.similarity_search("ページ１")[0].page_content == "ページ１\nページ１コンテンツ"