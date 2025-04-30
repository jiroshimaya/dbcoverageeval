"""
質問カバー率測定パイプラインのテストスクリプト
"""
import os
import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートディレクトリのパスを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# モジュールのインポート
from src.dbcoverageeval.ingest import ingest
from src.dbcoverageeval import __version__

def main():
    print(f"質問カバー率測定パイプライン テスト (バージョン: {__version__})")
    
    # 基本ディレクトリの設定
    base_dir = Path(__file__).parent
    docs_dir = base_dir / "docs"
    chunks_parquet = base_dir / "chunks.parquet"
    
    print(f"ドキュメントディレクトリ: {docs_dir}")
    print(f"出力Parquetファイル: {chunks_parquet}")
    
    # ディレクトリにあるファイルを表示
    print("\nドキュメントディレクトリの内容:")
    for file_path in docs_dir.glob("*"):
        print(f" - {file_path.name}")
    
    # ドキュメント取り込みの実行
    print("\nドキュメント取り込み処理を開始します...")
    try:
        ingest(
            docs_path=str(docs_dir),
            out_parquet=str(chunks_parquet),
            chunk_tokens=1000,
            chunk_overlap=200
        )
        
        # 結果の確認
        if chunks_parquet.exists():
            df = pd.read_parquet(chunks_parquet)
            print(f"\n取り込み完了: {len(df)} チャンクをParquetに保存しました")
            
            # ドキュメント数とページ数のカウント
            unique_docs = df["doc_id"].nunique()
            unique_pages = df["parent_id"].nunique()
            
            print(f"ドキュメント数: {unique_docs}")
            print(f"ページ数: {unique_pages}")
            print(f"チャンク数: {len(df)}")
            print(f"平均チャンク長: {df['text'].str.len().mean():.1f} 文字")
            print(f"最大チャンク長: {df['text'].str.len().max()} 文字")
            
            # 最初のチャンクの内容を表示
            print("\n最初のチャンクの内容:")
            first_chunk = df.iloc[0]
            print(f"チャンクID: {first_chunk['chunk_id']}")
            print(f"親ID: {first_chunk['parent_id']}")
            print(f"ドキュメントID: {first_chunk['doc_id']}")
            print(f"テキスト (先頭200文字): {first_chunk['text'][:200]}...")
        else:
            print("エラー: Parquetファイルが生成されませんでした")
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
