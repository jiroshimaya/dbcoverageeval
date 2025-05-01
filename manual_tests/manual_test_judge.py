from langchain_core.documents import Document

from dbcoverageeval.judge import JudgeResult, judge_search_result


def test_judge_search_result():
  
    # C > A > B > D
    all_documents = [
      Document(page_content="AはBよりも優れています。"),
      Document(page_content="CはDよりも優れています。"),
      Document(page_content="CはAより優れています。"),
      Document(page_content="BはDよりも優れています。"),
    ]
    question = "AとCはどちらが優れていますか？"
    documents = [
        Document(page_content="AはBよりも優れています。"),
    ]
    result_documents = [Document(page_content="ドキュメント")]
    judge_result = judge_search_result(question, documents)
    assert judge_result.result == "NG"
    