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


def generate_sample_pdf(filename: str = "sample.pdf", texts: list[str] = ["Sample", "Sample2"]) -> None:
  
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # フォント設定
    font_size = 20

    for text in texts:
      c.setFont("Helvetica", font_size)
      text_width = c.stringWidth(text, "Helvetica", font_size)
      c.drawString((width - text_width) / 2, height / 2, text)
      c.showPage()

    c.save()
    
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
    with tempfile.TemporaryDirectory() as temp_dir:
        doc_dir = os.path.join(temp_dir, "docs")
        os.makedirs(doc_dir, exist_ok=True)
        sample_path = os.path.join(doc_dir, "sample.pdf")
        generate_sample_pdf(filename=sample_path, texts=["sample 1", "sample 2"])
        sample2_path = os.path.join(doc_dir, "sample2.pdf")
        generate_sample_pdf(filename=sample2_path, texts=["sample 3", "sample 4"])
        db_path = os.path.join(temp_dir, "db")
        db = ingest(doc_dir, persist_dir=db_path)
        assert db is not None
        assert db.similarity_search("sample 1")[0].page_content == "sample 1"
        assert db.similarity_search("sample 2")[0].page_content == "sample 2"
        assert db.similarity_search("sample 3")[0].page_content == "sample 3"
        assert db.similarity_search("sample 4")[0].page_content == "sample 4"
