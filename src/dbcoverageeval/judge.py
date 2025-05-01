"""
判定モジュール。
質問とドキュメントから、カバー率を判定する。
"""
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

JUDGE_PROMPT_TEMPLATE = ChatPromptTemplate(
    [
        ("system", "あなたは厳密なレビュワーです。ドキュメント内に質問に対する十分な回答が含まれているかを評価してください。"),
        ("user", "質問: {question}\n\nドキュメント: {docs_text}"),        
    ]
)

class JudgeResult(BaseModel, ):
    """判定結果"""
    reason: str = Field(description="判定理由")
    judge: Literal["OK", "NG", "Partially"] = Field(description="判定結果")

class DetailedJudgeResult(JudgeResult):
    """詳細な判定結果"""
    question: str = Field(description="質問")
    search_results: list[Document] = Field(description="ドキュメントのID")

def judge_search_result(question: str, documents: list[Document]) -> JudgeResult:
    """質問とドキュメントから判定結果を返す"""
    
    model = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(JudgeResult)
    
    
    chain = JUDGE_PROMPT_TEMPLATE | model
    
    result = chain.invoke({"question": question, "docs_text": "\n".join([doc.page_content for doc in documents])})
    
    detailed_result = DetailedJudgeResult(question=question, judge=result.judge, reason=result.reason, search_results=documents)
    
    return detailed_result
