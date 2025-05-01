"""
PDFダウンロードテストスクリプト
"""
import os

import requests


def download_pdf(url: str, output_dir: str = os.path.join(os.path.dirname(__file__))) -> str:
    """
    PDFファイルをダウンロードする

    Args:
        url: ダウンロードするPDFのURL
        output_dir: 保存先ディレクトリ

    Returns:
        ダウンロードしたファイルのパス
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # URLからファイル名を取得
    filename = url.split('/')[-1]
    output_path = os.path.join(output_dir, filename)
    
    print(f"ダウンロードを開始します: {url}")
    
    # ファイルのダウンロード
    response = requests.get(url)
    response.raise_for_status()  # エラーチェック
    
    # ファイルの保存
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    return output_path


if __name__ == "__main__":
    # ダウンロードするPDFのURL
    url = "https://www.city.kobe.lg.jp/documents/15123/r4_doukou.pdf"

    try:
        output_path = download_pdf(url)
        print(f"ダウンロードが完了しました: {output_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
