from dbcoverageeval.evaluate import evaluate


def main():
    docs_path = "manual_tests/data"
    questions = ["AとBはどちらが優れていますか？", "CとDはどちらが優れていますか？"]
    report = evaluate(docs_path, questions, "report.json")
    print(report)

if __name__ == "__main__":
    main()
