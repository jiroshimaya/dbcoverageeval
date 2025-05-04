import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


class SearchEvaluation(BaseModel):
    """検索結果の評価"""
    explanation: str = Field(description="評価の理由")
    label: Literal["OK", "Partial", "NG"] = Field(description="検索結果の評価ラベル")

# プロンプトテンプレート
EVAL_PROMPT_TEMPLATE = ChatPromptTemplate(
    [
        ("system", "あなたは検索結果の評価を行うアシスタントです。質問に対する検索結果が、その質問に答えるための十分な情報を含んでいるかを評価してください。"),
        ("user", """質問: {query}

検索結果:
{search_results}

上記の検索結果は、質問に答えるための十分な情報を含んでいますか？
評価の理由を出力してから以下の3つのラベルのいずれかで回答してください：
- OK: 質問に完全に答えるための情報が含まれている
- Partial: 質問に部分的に答えるための情報が含まれている
- NG: 質問に答えるための情報が含まれていない"""),
    ]
)

# 設定
CACHE_DIR = Path("cache")
TOP_K = 10  # 各クエリについて評価する検索結果の数
NUM_SAMPLES = -1  # 評価するクエリの数
BATCH_SIZE = 10  # バッチサイズ

def get_search_results(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus: list[dict],
    k: int = 10
) -> list[list[dict]]:
    """各クエリの検索結果を取得

    Args:
        query_embeddings: クエリの埋め込みベクトル。shape=(クエリ数, 埋め込み次元)
        corpus_embeddings: コーパスの埋め込みベクトル。shape=(コーパス数, 埋め込み次元)
        corpus: コーパスのリスト。各要素は以下のキーを含む辞書:
            - docid: 文書ID
            - text: 文書のテキスト
            - title: 文書のタイトル（オプション）
        k: 各クエリについて取得する検索結果の数

    Returns:
        各クエリの検索結果のリスト。shape=(クエリ数, k)
        各検索結果は元のコーパスの辞書と同じ形式
    """
    # コサイン類似度を計算
    similarities = query_embeddings @ corpus_embeddings.T
    
    # 各クエリについて上位k件を取得
    results = []
    for i in range(len(query_embeddings)):
        top_k_indices = np.argsort(-similarities[i])[:k]
        results.append([corpus[idx] for idx in top_k_indices])
    
    return results

def create_positive_sample(
    query: dict,
    sorted_indices: np.ndarray,
    pos_indices: list[int],
    corpus: list[dict],
    k: int
) -> dict:
    """正解を含む検索結果を作成

    Args:
        query: 質問。以下のキーを含む辞書:
            - query: 質問文
            - positive_passages: 正解文書のリスト
        sorted_indices: 類似度でソートされた文書のインデックス
        pos_indices: 正解文書のインデックスのリスト
        corpus: コーパスのリスト
        k: 検索結果に含める文書数

    Returns:
        評価用のサンプル。以下のキーを含む辞書:
            - query: 質問文
            - search_results: 検索結果の文書リスト
            - has_positive: 正解文書を含むかどうか
            - positive_passages: 正解文書リスト
    """
    top_k_indices = sorted_indices[:k].tolist()
    # sorted_indicesの順序で正解文書を取得
    missing_positives = [idx for idx in sorted_indices if idx in pos_indices and idx not in top_k_indices]
    
    if missing_positives:
        # 含まれていない正解を、正解でない文書と入れ替え
        # 後ろから順に処理することで、順序を維持
        for pos_idx in reversed(missing_positives):
            # 後ろから順に正解でない文書を探す
            for idx in reversed(top_k_indices):
                if idx not in pos_indices:
                    # 正解でない文書を見つけたら、それを正解と入れ替え
                    replace_idx = top_k_indices.index(idx)
                    top_k_indices[replace_idx] = pos_idx
                    break
    
    results = [corpus[idx] for idx in top_k_indices]
    return {
        "query": query["query"],
        "search_results": results,
        "has_positive": True,
        "positive_passages": query["positive_passages"]
    }

def create_negative_sample(
    query: dict,
    sorted_indices: np.ndarray,
    pos_indices: list[int],
    corpus: list[dict],
    k: int
) -> dict:
    """正解を含まない検索結果を作成

    Args:
        query: 質問。以下のキーを含む辞書:
            - query: 質問文
            - positive_passages: 正解文書のリスト
        sorted_indices: 類似度でソートされた文書のインデックス
        pos_indices: 正解文書のインデックスのリスト
        corpus: コーパスのリスト
        k: 検索結果に含める文書数

    Returns:
        評価用のサンプル。以下のキーを含む辞書:
            - query: 質問文
            - search_results: 検索結果の文書リスト
            - has_positive: 正解文書を含むかどうか
            - positive_passages: 正解文書リスト
    """
    # sorted_indicesから正解でないものをk個取得
    neg_indices = []
    for idx in sorted_indices:
        if idx not in pos_indices:
            neg_indices.append(idx)
            if len(neg_indices) == k:
                break
    
    results = [corpus[idx] for idx in neg_indices]
    return {
        "query": query["query"],
        "search_results": results,
        "has_positive": False,
        "positive_passages": query["positive_passages"]
    }

def create_positive_negative_samples(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus: list[dict],
    queries: list[dict],
    positive_indices: list[list[int]],
    k: int = 10
) -> list[dict]:
    """各クエリに対して正解を含む/含まない検索結果を作成

    Args:
        query_embeddings: クエリの埋め込みベクトル。shape=(クエリ数, 埋め込み次元)
        corpus_embeddings: コーパスの埋め込みベクトル。shape=(コーパス数, 埋め込み次元)
        corpus: コーパスのリスト。各要素は以下のキーを含む辞書:
            - docid: 文書ID
            - text: 文書のテキスト
            - title: 文書のタイトル（オプション）
        queries: 質問のリスト。各質問は以下のキーを含む辞書:
            - query: 質問文
            - positive_passages: 正解文書のリスト
        positive_indices: 各質問の正解文書のインデックスのリスト
        k: 各検索結果に含める文書数

    Returns:
        評価用のサンプルのリスト。各サンプルは以下のキーを含む辞書:
            - query: 質問文
            - search_results: 検索結果の文書リスト
            - has_positive: 正解文書を含むかどうか
            - positive_passages: 正解文書リスト
    """
    samples = []
    
    # コサイン類似度を計算
    similarities = query_embeddings @ corpus_embeddings.T
    
    for i, (query, pos_indices) in enumerate(zip(queries, positive_indices)):
        # 類似度でソートした全インデックス
        sorted_indices = np.argsort(-similarities[i])
        
        # 正解を含む検索結果を作成
        if pos_indices:  # 正解文書が存在する場合
            samples.append(create_positive_sample(query, sorted_indices, pos_indices, corpus, k))
        
        # 正解を含まない検索結果を作成
        samples.append(create_negative_sample(query, sorted_indices, pos_indices, corpus, k))
    
    return samples

def evaluate_with_llm(samples: list[dict]) -> list[dict[str, Any]]:
    """LLMで評価を実行

    Args:
        samples: 評価用のサンプルのリスト。各サンプルは以下のキーを含む辞書:
            - query: 質問文
            - search_results: 検索結果の文書リスト
            - has_positive: 正解文書を含むかどうか
            - positive_passages: 正解文書リスト

    Returns:
        評価結果のリスト。各結果は以下のキーを含む辞書:
            - query: 質問文
            - has_positive: 正解文書を含むかどうか
            - llm_label: LLMによる評価ラベル（OK/Partial/NG）
            - explanation: 評価の理由
    """
    # LLMの設定
    model = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(SearchEvaluation)
    chain = EVAL_PROMPT_TEMPLATE | model
    
    # 入力データの準備
    inputs = []
    for sample in samples:
        search_results_text = ""
        for i, result in enumerate(sample["search_results"], 1):
            title = result.get("title", "")
            title_text = f"【{title}】" if title else ""
            search_results_text += f"\n{i}. {title_text}{result['text']}..."
        
        inputs.append({
            "query": sample["query"],
            "search_results": search_results_text
        })
    
    # バッチ処理で評価を実行
    with tqdm(total=len(samples), desc="LLMで評価中") as pbar:
        eval_results = []
        for i in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[i:i + BATCH_SIZE]
            batch_results = chain.batch(batch)
            eval_results.extend(batch_results)
            pbar.update(len(batch))
    
    # 結果の整形
    results = []
    for sample, eval_result in zip(samples, eval_results):
        results.append({
            "query": sample["query"],
            "has_positive": sample["has_positive"],
            "llm_label": eval_result.label,
            "explanation": eval_result.explanation
        })
    
    return results

def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """評価指標を計算

    Args:
        results: evaluate_with_llmの戻り値。各結果は以下のキーを含む辞書:
            - has_positive: 正解文書を含むかどうか
            - llm_label: LLMによる評価ラベル（OK/Partial/NG）

    Returns:
        評価指標を含む辞書:
            - positive_cases: 正解を含むケースの数
            - negative_cases: 正解を含まないケースの数
            - positive_rates: 正解を含むケースのラベル分布
            - negative_rates: 正解を含まないケースのラベル分布
    """
    # 正解を含むケースと含まないケースを分ける
    positive_cases = [r for r in results if r["has_positive"]]
    negative_cases = [r for r in results if not r["has_positive"]]
    
    # 各ケースでのラベル分布を計算
    def calc_label_rates(cases: list[dict[str, Any]]) -> dict[str, float]:
        """各ラベルの出現率を計算

        Args:
            cases: 評価結果のリスト。各要素は以下のキーを含む辞書:
                - llm_label: LLMによる評価ラベル（OK/Partial/NG）

        Returns:
            各ラベルの出現率を含む辞書:
                - OK: OKラベルの割合
                - Partial: Partialラベルの割合
                - NG: NGラベルの割合
        """
        total = len(cases)
        if total == 0:
            return {"OK": 0.0, "Partial": 0.0, "NG": 0.0}
        
        return {
            "OK": sum(1 for r in cases if r["llm_label"] == "OK") / total,
            "Partial": sum(1 for r in cases if r["llm_label"] == "Partial") / total,
            "NG": sum(1 for r in cases if r["llm_label"] == "NG") / total
        }
    
    positive_rates = calc_label_rates(positive_cases)
    negative_rates = calc_label_rates(negative_cases)
    
    return {
        "positive_cases": len(positive_cases),
        "negative_cases": len(negative_cases),
        "positive_rates": positive_rates,
        "negative_rates": negative_rates
    }

def save_evaluation_results(
    results: List[dict],
    metrics: Dict[str, Any],
    samples: List[dict],
    output_dir: Path
) -> None:
    """評価結果をJSONファイルとして保存

    Args:
        results: evaluate_with_llmの結果
        metrics: calculate_metricsの結果
        samples: 評価に使用したサンプル
        output_dir: 出力先ディレクトリ
    """
    # ドキュメントの辞書を作成
    documents = {}
    for sample in samples:
        for doc in sample["search_results"]:
            if doc["docid"] not in documents:
                documents[doc["docid"]] = doc["text"]
    
    # 詳細な評価結果を作成
    details = []
    for result, sample in zip(results, samples):
        details.append({
            "query": result["query"],
            "has_positive": result["has_positive"],
            "llm_label": result["llm_label"],
            "explanation": result["explanation"],
            "search_result_ids": [doc["docid"] for doc in sample["search_results"]],
            "positive_passage_ids": [doc["docid"] for doc in sample["positive_passages"]]
        })
    
    # 評価結果全体を作成
    evaluation_results = {
        "summary": {
            "total_samples": len(results),
            "positive_cases": metrics["positive_cases"],
            "positive_rates": metrics["positive_rates"],
            "negative_cases": metrics["negative_cases"],
            "negative_rates": metrics["negative_rates"]
        },
        "details": details,
        "documents": documents
    }
    
    # 出力先ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSONファイルとして保存
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n評価結果を保存しました: {output_file}")

def main():
    # キャッシュされた埋め込みを読み込み
    print("埋め込みを読み込み中...")
    with open(CACHE_DIR / "embeddings_queries.pkl", "rb") as f:
        query_embeddings = pickle.load(f)
    with open(CACHE_DIR / "embeddings_corpus.pkl", "rb") as f:
        corpus_embeddings = pickle.load(f)
    
    # データセットを読み込み
    print("データセットを読み込み中...")
    with open(CACHE_DIR / "sample_queries_100.pkl", "rb") as f:
        queries = pickle.load(f)
    with open(CACHE_DIR / "sample_corpus_10000.pkl", "rb") as f:
        corpus = list(pickle.load(f))
    
    # 評価するクエリをランダムに選択
    if NUM_SAMPLES > 0 and NUM_SAMPLES < len(queries):
        print(f"ランダムに{NUM_SAMPLES}個のクエリを選択します...")
        selected_indices = random.sample(range(len(queries)), NUM_SAMPLES)
        queries = [queries[i] for i in selected_indices]
        query_embeddings = query_embeddings[selected_indices]
    
    all_docids = [doc["docid"] for doc in corpus]
    
    # 正解文書のインデックスを取得
    positive_indices = []
    for query in queries:
        pos_docids = [p["docid"] for p in query["positive_passages"]]
        pos_indices = [
            i for i, docid in enumerate(all_docids)
            if docid in pos_docids
        ]
        positive_indices.append(pos_indices)
    
    # 評価用サンプルを作成
    print("評価用サンプルを作成中...")
    samples = create_positive_negative_samples(
        query_embeddings,
        corpus_embeddings,
        corpus,
        queries,
        positive_indices,
        k=TOP_K
    )
    
    # LLMで評価
    evaluation_results = evaluate_with_llm(samples)
    
    # 評価指標を計算
    metrics = calculate_metrics(evaluation_results)
    
    # 評価結果を表示
    print("\n=== 評価結果 ===")
    print(f"評価サンプル数: {len(evaluation_results)}")
    print(f"正解を含むケース: {metrics['positive_cases']}")
    print("- ラベル分布:")
    for label, rate in metrics["positive_rates"].items():
        print(f"  - {label}: {rate:.3f}")
    
    print(f"\n正解を含まないケース: {metrics['negative_cases']}")
    print("- ラベル分布:")
    for label, rate in metrics["negative_rates"].items():
        print(f"  - {label}: {rate:.3f}")
    
    # 評価結果をファイルに保存
    save_evaluation_results(
        evaluation_results,
        metrics,
        samples,
        output_dir=CACHE_DIR / "results"
    )

if __name__ == "__main__":
    main() 