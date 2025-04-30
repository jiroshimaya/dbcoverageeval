"""
テキストファイルをPDFファイルに変換するユーティリティスクリプト
"""
import os
from fpdf import FPDF
import textwrap

def text_to_pdf(text_file_path, pdf_file_path):
    """
    テキストファイルをPDFに変換する
    
    Args:
        text_file_path: 入力テキストファイルのパス
        pdf_file_path: 出力PDFファイルのパス
    """
    # テキストファイルを読み込む
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # PDFを作成
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)
    
    # テキストをPDFに追加
    # 行ごとに分割
    lines = text.split('\n')
    for line in lines:
        # 見出し（#で始まる行）の処理
        if line.startswith('# '):
            pdf.set_font('DejaVu', '', 18)
            pdf.cell(0, 10, line[2:], ln=True)
            pdf.set_font('DejaVu', '', 12)
        # サブ見出し（##で始まる行）の処理
        elif line.startswith('## '):
            pdf.set_font('DejaVu', '', 16)
            pdf.cell(0, 10, line[3:], ln=True)
            pdf.set_font('DejaVu', '', 12)
        # サブサブ見出し（###で始まる行）の処理
        elif line.startswith('### '):
            pdf.set_font('DejaVu', '', 14)
            pdf.cell(0, 10, line[4:], ln=True)
            pdf.set_font('DejaVu', '', 12)
        # リスト項目（- で始まる行）の処理
        elif line.strip().startswith('- '):
            # インデントを追加
            indented_line = '   • ' + line.strip()[2:]
            # 長い行は折り返す
            wrapped_lines = textwrap.wrap(indented_line, width=80)
            for wrapped_line in wrapped_lines:
                pdf.cell(0, 10, wrapped_line, ln=True)
        # 数字付きリスト項目の処理
        elif line.strip() and line.strip()[0].isdigit() and '. ' in line.strip():
            # 数字とピリオドを保持して残りをラップ
            parts = line.strip().split('. ', 1)
            indented_line = '   ' + parts[0] + '. ' + parts[1]
            wrapped_lines = textwrap.wrap(indented_line, width=80)
            for wrapped_line in wrapped_lines:
                pdf.cell(0, 10, wrapped_line, ln=True)
        # 空行の処理
        elif not line.strip():
            pdf.ln(10)
        # 通常のテキスト行
        else:
            # 長い行は折り返す
            wrapped_lines = textwrap.wrap(line, width=80)
            for wrapped_line in wrapped_lines:
                pdf.cell(0, 10, wrapped_line, ln=True)
    
    # PDFを保存
    pdf.output(pdf_file_path)

def main():
    # 入力テキストファイルと出力PDFファイルの指定
    base_dir = '/Users/jiro/development/dbcoverageeval/test_data'
    docs_dir = os.path.join(base_dir, 'docs')
    
    # 出力ディレクトリが存在しなければ作成
    os.makedirs(docs_dir, exist_ok=True)
    
    # テキストをPDFに変換
    text_to_pdf(
        os.path.join(base_dir, 'ai_basics.txt'),
        os.path.join(docs_dir, 'ai_basics.pdf')
    )
    
    text_to_pdf(
        os.path.join(base_dir, 'data_science.txt'),
        os.path.join(docs_dir, 'data_science.pdf')
    )
    
    print("PDFファイルの作成が完了しました。")

if __name__ == "__main__":
    main()
