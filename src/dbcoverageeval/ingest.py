"""
ドキュメント取り込みモジュール。
各種形式のドキュメントからテキストを抽出し、チャンク化して保存する。
"""
import mimetypes
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def load_pdf_files(docs_path: str | Path) -> List[Document]:
    """
    ドキュメントをロードする
    """
    if isinstance(docs_path, str):
        docs_path = Path(docs_path)
    if not docs_path.exists():
        raise FileNotFoundError(f"File not found: {docs_path}")
    # docs_pathの直下にあるpdfを取得
    pdf_files = list(docs_path.glob('**/*.pdf'))
    
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        documents.extend(docs)
    for document in documents:
        document.metadata["id"] = str(uuid.uuid4())
    return documents

def ingest_documents(documents: List[Document]) -> Chroma:
    db = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings()
    )
    return db

def ingest(docs_path: str | Path) -> Chroma:
    """
    ドキュメントを取り込み、DBに保存する
    
    Args:
        docs_path: ドキュメントが格納されているディレクトリのパス
        output_path: 出力するDBのパス
    """
    docs_path = Path(docs_path)
    
    documents = load_pdf_files(docs_path)
    db = ingest_documents(documents)
    
    return db
