import tqdm

from dbcoverageeval.ingest import ingest
from dbcoverageeval.judge import judge_search_result
from dbcoverageeval.report import Report, calculate_metrics


def evaluate(docs_path: str, questions: list[str], output_path: str = ""):
    print("ingest")
    db = ingest(docs_path, initialize=False)
    results = []
    for question in tqdm.tqdm(questions):
        documents = db.similarity_search(question)
        result = judge_search_result(question, documents)
        results.append(result)
    report = calculate_metrics(results)
    if output_path:
        report.save(output_path)
    return report
