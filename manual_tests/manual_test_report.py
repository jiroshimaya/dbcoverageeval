from langchain_core.documents import Document

from dbcoverageeval.judge import DetailedJudgeResult
from dbcoverageeval.report import Report, calculate_metrics


def test_calculate_metrics():
  
    # C > A > B > D
    all_documents = [
      Document(page_content="AはBよりも優れています。", metadata={"id": "1"}),
      Document(page_content="CはDよりも優れています。", metadata={"id": "2"}),
      Document(page_content="CはAより優れています。", metadata={"id": "3"}),
      Document(page_content="BはDよりも優れています。", metadata={"id": "4"}),
    ]

    results = [
      # NG
      DetailedJudgeResult(question="AとCはどちらが優れていますか？", judge="NG", reason="AはCよりも優れています。", search_results=[Document(page_content="AはBよりも優れています。", metadata={"id": "1"}), Document(page_content="CはDよりも優れています。", metadata={"id": "2"})]),
      # OK
      DetailedJudgeResult(question="AとBはどちらが優れていますか？", judge="OK", reason="AはBよりも優れています。", search_results=[Document(page_content="AはBよりも優れています。", metadata={"id": "1"}), Document(page_content="CはDよりも優れています。", metadata={"id": "2"})]),
    ]
    
    report = calculate_metrics(results)
    
    assert report.summary["count"]["OK"] == 1
    assert report.summary["count"]["NG"] == 1
    assert report.summary["count"]["total"] == 2
    assert report.summary["ratio"]["OK"] == 0.5
    assert report.summary["ratio"]["NG"] == 0.5
    
    
