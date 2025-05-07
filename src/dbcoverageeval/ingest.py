"""
ドキュメント取り込みモジュール。
各種形式のドキュメントからテキストを抽出し、チャンク化して保存する。
"""

import uuid
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def load_pdf_files(docs_path: str | Path) -> List[Document]:
    """
    ドキュメントをロードする
    """
    if isinstance(docs_path, str):
        docs_path = Path(docs_path)
    if not docs_path.exists():
        raise FileNotFoundError(f"File not found: {docs_path}")
    # docs_pathの直下にあるpdfを取得
    pdf_files = list(docs_path.glob("**/*.pdf"))

    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        documents.extend(docs)
    for document in documents:
        document.metadata["id"] = str(uuid.uuid4())
    return documents


def load_csv_files(
    docs_path: str | Path, source_column: str, metadata_columns: List[str]
) -> List[Document]:
    """
    ドキュメントをロードする
    """
    if isinstance(docs_path, str):
        docs_path = Path(docs_path)
    if not docs_path.exists():
        raise FileNotFoundError(f"File not found: {docs_path}")
    # docs_pathの直下にあるcsvを取得
    csv_files = list(docs_path.glob("**/*.csv"))
    documents = []
    for csv_file in csv_files:
        docs = CSVLoader(
            csv_file, source_column=source_column, metadata_columns=metadata_columns
        ).load()
        documents.extend(docs)
    return documents


def load_or_create_db(persist_dir: str) -> Chroma:
    """
    データベースをロードするか、新しく作成する
    """
    db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=persist_dir)
    return db


def add_documents(db: Chroma, documents: List[Document]) -> Chroma:
    """
    ドキュメントを新しいものだけ追加する
    """
    all_documents = set(db._collection.get()["documents"])
    new_documents = [doc for doc in documents if doc.page_content not in all_documents]

    if new_documents:
        db.add_documents(documents=new_documents)
    return db


def reset_db(db: Chroma) -> Chroma:
    """
    データベースをリセットする
    """
    db.reset_collection()
    return db


def ingest(
    docs_path: str | Path, initialize: bool = False, persist_dir: str = "./chroma_db"
) -> Chroma:
    """
    ドキュメントを取り込み、DBに保存する

    Args:
        docs_path: ドキュメントが格納されているディレクトリのパス
        output_path: 出力するDBのパス
    """
    docs_path = Path(docs_path)

    documents = load_pdf_files(docs_path)
    db = load_or_create_db(persist_dir)
    if initialize:
        db = reset_db(db)
    db = add_documents(db, documents)

    return db
