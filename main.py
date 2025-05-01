import json

from dbcoverageeval.evaluate import evaluate


def main():
    docs_path = "manual_tests/data/docs"
    question_path = "manual_tests/data/questions.json"
    with open(question_path, "r") as f:
        question_json = json.load(f)
    questions = [question["question"] for question in question_json]
    report = evaluate(docs_path, questions, "report.json")
    print(report)

if __name__ == "__main__":
    main()
