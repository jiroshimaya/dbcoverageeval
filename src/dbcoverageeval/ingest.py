"""
ドキュメント取り込みモジュール。
各種形式のドキュメントからテキストを抽出し、チャンク化して保存する。
"""
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import tiktoken
import mimetypes
from abc import ABC, abstractmethod

# 抽出器の抽象クラス
class Extractor(ABC):
    """ドキュメント抽出の基底クラス。新しい形式に対応する際はこれを継承する。"""
    
    @abstractmethod
    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """
        ファイルからテキストを抽出する
        
        Args:
            file_path: 抽出対象ファイルのパス
            
        Returns:
            抽出されたページごとのテキスト。各ページは以下の形式:
            {
                "text": str,
                "page_num": int,
                "doc_id": str
            }
        """
        pass

# テキストファイル抽出器
class TextExtractor(Extractor):
    """テキストファイル用抽出器"""
    
    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """テキストファイルから内容を抽出"""
        doc_id = os.path.basename(file_path).split('.')[0]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 1ページとして扱う
            return [{
                "text": content,
                "page_num": 1,
                "doc_id": doc_id
            }]
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return []

# PDF抽出器
class PDFExtractor(Extractor):
    """PDFファイル用抽出器"""
    
    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """PDFからテキストを抽出"""
        import pdfplumber
        doc_id = os.path.basename(file_path).split('.')[0]
        result = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        result.append({
                            "text": text,
                            "page_num": i + 1,
                            "doc_id": doc_id
                        })
            return result
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return []

# DOCX抽出器
class DocxExtractor(Extractor):
    """DOCXファイル用抽出器"""
    
    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """DOCXからテキストを抽出"""
        import docx
        doc_id = os.path.basename(file_path).split('.')[0]
        
        try:
            doc = docx.Document(file_path)
            # DOCXはページ区切り情報がないため、全体を1ページとして扱う
            text = "\n".join([p.text for p in doc.paragraphs])
            return [{
                "text": text,
                "page_num": 1,
                "doc_id": doc_id
            }]
        except Exception as e:
            print(f"Error extracting text from DOCX {file_path}: {e}")
            return []

# Excel抽出器
class ExcelExtractor(Extractor):
    """Excel用抽出器"""
    
    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Excelからテキストを抽出"""
        import pandas as pd
        doc_id = os.path.basename(file_path).split('.')[0]
        result = []
        
        try:
            # Excelのすべてのシートを読み込む
            excel_file = pd.ExcelFile(file_path)
            for i, sheet_name in enumerate(excel_file.sheet_names):
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                # DataFrameをテキストに変換
                text = df.to_string()
                result.append({
                    "text": text,
                    "page_num": i + 1,  # シート番号をページ番号として扱う
                    "doc_id": doc_id
                })
            return result
        except Exception as e:
            print(f"Error extracting text from Excel {file_path}: {e}")
            return []

def get_extractor(file_path: str) -> Optional[Extractor]:
    """
    ファイルの種類に適した抽出器を返す
    
    Args:
        file_path: ファイルパス
    
    Returns:
        適切な抽出器、またはNone（未対応の形式）
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    
    if mime_type == 'text/plain' or ext in ['.txt', '.md']:
        return TextExtractor()
    elif mime_type == 'application/pdf' or ext == '.pdf':
        return PDFExtractor()
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or ext == '.docx':
        return DocxExtractor()
    elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] or ext in ['.xls', '.xlsx']:
        return ExcelExtractor()
    else:
        print(f"Unsupported file type: {mime_type} ({file_path})")
        return None

def chunk_text(text: str, chunk_tokens: int = 1000, overlap_tokens: int = 200) -> List[str]:
    """
    テキストをトークン数に基づいてチャンク化する
    
    Args:
        text: チャンク化するテキスト
        chunk_tokens: 1チャンクのトークン最大数
        overlap_tokens: チャンク間のオーバーラップトークン数
    
    Returns:
        チャンク化されたテキストのリスト
    """
    # OpenAIのエンコーダーを使用
    tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4対応のエンコーダー
    tokens = tokenizer.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        # チャンクのトークン範囲を計算
        chunk_end = min(i + chunk_tokens, len(tokens))
        chunk = tokenizer.decode(tokens[i:chunk_end])
        chunks.append(chunk)
        
        # オーバーラップを考慮して次のチャンクの開始位置を計算
        i = chunk_end - overlap_tokens if chunk_end < len(tokens) else len(tokens)
    
    return chunks

def ingest(docs_path: str, out_parquet: str, chunk_tokens: int = 1000, chunk_overlap: int = 200) -> None:
    """
    ドキュメントを取り込み、チャンク化してParquetに保存する
    
    Args:
        docs_path: ドキュメントが格納されているディレクトリのパス
        out_parquet: 出力するParquetファイルのパス
        chunk_tokens: 1チャンクのトークン最大数
        chunk_overlap: チャンク間のオーバーラップトークン数
    """
    all_chunks = []
    docs_path = Path(docs_path)
    
    # ディレクトリ内のすべてのファイルを処理
    for file_path in docs_path.glob('**/*'):
        if file_path.is_file():
            extractor = get_extractor(str(file_path))
            if extractor:
                pages = extractor.extract(str(file_path))
                
                for page in pages:
                    doc_id = page["doc_id"]
                    page_num = page["page_num"]
                    parent_id = f"{doc_id}_p{page_num}"
                    
                    # テキストをチャンク化
                    text_chunks = chunk_text(
                        page["text"], 
                        chunk_tokens=chunk_tokens, 
                        overlap_tokens=chunk_overlap
                    )
                    
                    # チャンク情報を保存
                    for i, chunk_text in enumerate(text_chunks):
                        all_chunks.append({
                            "chunk_id": f"{parent_id}_c{i+1}",
                            "parent_id": parent_id,
                            "doc_id": doc_id,
                            "page_num": page_num,
                            "chunk_num": i + 1,
                            "text": chunk_text
                        })
    
    # チャンクをDataFrameに変換してParquetに保存
    if all_chunks:
        df = pd.DataFrame(all_chunks)
        df.to_parquet(out_parquet, index=False)
        print(f"Saved {len(all_chunks)} chunks to {out_parquet}")
    else:
        print("No chunks were generated. Check your documents and extractors.")
