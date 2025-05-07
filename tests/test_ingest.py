import os
import tempfile


def generate_sample_pdf(
    filename: str = "sample.pdf", texts: list[str] = ["Sample", "Sample2"]
) -> None:
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


def generate_sample_csv(
    filename: str = "sample.csv", texts: list[str] = ["Sample", "Sample2"]
) -> None:
    csv_text = "id,text,category\n1,sample 1,cat1\n2,sample 2,cat2\n"
    with open(filename, "w") as f:
        f.write(csv_text)


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


def test_load_csv_files():
    from dbcoverageeval.ingest import load_csv_files

    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        # サンプルCSVを生成
        sample_path = os.path.join(temp_dir, "sample.csv")
        generate_sample_csv(filename=sample_path, texts=["sample 1", "sample 2"])

        documents = load_csv_files(
            temp_dir, source_column=None, metadata_columns=["category"]
        )
        assert len(documents) == 2
        assert documents[0].page_content == "id: 1\ntext: sample 1"
        assert documents[0].metadata["category"] == "cat1"
        assert documents[0].metadata["source"] == sample_path
        assert documents[1].page_content == "id: 2\ntext: sample 2"
        assert documents[1].metadata["category"] == "cat2"
        assert documents[1].metadata["source"] == sample_path
