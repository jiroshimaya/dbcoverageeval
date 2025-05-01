import os
import tempfile

import pytest


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

def test_load_pdf_files():
    from dbcoverageeval.ingest import load_pdf_files

    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        # サンプルPDFを生成
        sample_path = os.path.join(temp_dir, "sample.pdf")
        generate_sample_pdf(filename=sample_path, texts=["sample 1", "sample 2"])
        
        documents = load_pdf_files(temp_dir)
        assert len(documents) == 2
        assert documents[0].page_content == "sample 1"
        assert documents[1].page_content == "sample 2"
        
        
        
  
    
