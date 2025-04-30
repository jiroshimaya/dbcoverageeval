"""
pytest設定ファイル。
テスト用のフィクスチャとヘルパー関数を提供する。
"""
import os
import tempfile
import pytest
import pandas as pd
import yaml
from pathlib import Path


@pytest.fixture
def temp_dir():
    """一時ディレクトリを提供するフィクスチャ"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def sample_config():
    """テスト用の設定を提供するフィクスチャ"""
    return {
        "provider": "openai",
        "models": {
            "embedding": "text-embedding-3-small",
            "query_gen": "gpt-4o-mini",
            "judge": "gpt-4o"
        },
        "openai": {
            "api_key": "test_api_key",
            "api_base": "https://api.openai.com/v1"
        },
        "concurrency": 2,
        "chunk_tokens": 500,
        "chunk_overlap": 100,
        "top_k_child": 5,
        "top_k_parent": 10,
        "max_parents_for_judge": 2
    }


@pytest.fixture
def config_file(temp_dir, sample_config):
    """テスト用の設定ファイルを作成するフィクスチャ"""
    config_path = os.path.join(temp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_chunks():
    """テスト用のテキストチャンクを提供するフィクスチャ"""
    return [
        {
            "chunk_id": "doc1_p1_c1",
            "parent_id": "doc1_p1",
            "doc_id": "doc1",
            "page_num": 1,
            "chunk_num": 1,
            "text": "これはテスト用のドキュメント1の1ページ目のチャンク1です。質問Aに関する情報が含まれています。"
        },
        {
            "chunk_id": "doc1_p1_c2",
            "parent_id": "doc1_p1",
            "doc_id": "doc1",
            "page_num": 1,
            "chunk_num": 2,
            "text": "これはテスト用のドキュメント1の1ページ目のチャンク2です。質問Bに関する部分的な情報が含まれています。"
        },
        {
            "chunk_id": "doc2_p1_c1",
            "parent_id": "doc2_p1",
            "doc_id": "doc2",
            "page_num": 1,
            "chunk_num": 1,
            "text": "これはテスト用のドキュメント2の1ページ目のチャンク1です。質問Cに関する情報は含まれていません。"
        }
    ]


@pytest.fixture
def chunks_parquet(temp_dir, sample_chunks):
    """テスト用のParquetファイルを作成するフィクスチャ"""
    parquet_path = os.path.join(temp_dir, "chunks.parquet")
    df = pd.DataFrame(sample_chunks)
    df.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture
def sample_questions():
    """テスト用の質問リストを提供するフィクスチャ"""
    return [
        {"id": "q_001", "question": "質問A"},
        {"id": "q_002", "question": "質問B"},
        {"id": "q_003", "question": "質問C"}
    ]


@pytest.fixture
def questions_tsv(temp_dir, sample_questions):
    """テスト用の質問TSVファイルを作成するフィクスチャ"""
    tsv_path = os.path.join(temp_dir, "questions.tsv")
    df = pd.DataFrame(sample_questions)
    df.to_csv(tsv_path, sep="\t", index=False)
    return tsv_path


@pytest.fixture
def questions_csv(temp_dir, sample_questions):
    """テスト用の質問CSVファイルを作成するフィクスチャ"""
    csv_path = os.path.join(temp_dir, "questions.csv")
    df = pd.DataFrame(sample_questions)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_openai_embeddings():
    """OpenAIのエンベディングAPIをモックするフィクスチャ"""
    # 実際のAPIを呼び出さないようにモック化
    import openai
    original_create = openai.embeddings.create

    def mock_create(*args, **kwargs):
        # モックのレスポンスを作成
        class MockData:
            def __init__(self, embedding):
                self.embedding = embedding

        class MockResponse:
            def __init__(self, data):
                self.data = data

        # ダミーの埋め込みベクトルを生成
        input_texts = kwargs["input"]
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        data = []
        for i, _ in enumerate(input_texts):
            # 128次元のダミー埋め込みベクトル
            embedding = [0.1] * 128
            embedding[0] = 0.5 + 0.1 * i  # 各テキストで少し異なる値にする
            data.append(MockData(embedding))

        return MockResponse(data)

    # モック関数を設定
    openai.embeddings.create = mock_create
    yield
    # テスト後に元の関数を復元
    openai.embeddings.create = original_create


@pytest.fixture
def mock_openai_chat_completions():
    """OpenAIのチャット完了APIをモックするフィクスチャ"""
    import openai
    original_create = openai.chat.completions.create

    def mock_create(*args, **kwargs):
        # モックのレスポンスを作成
        class MockChoice:
            def __init__(self, message):
                self.message = message

        class MockMessage:
            def __init__(self, content):
                self.content = content

        class MockResponse:
            def __init__(self, choices):
                self.choices = choices

        # 質問に基づいてレスポンスを生成
        messages = kwargs["messages"]
        content = ""
        
        for msg in messages:
            if msg["role"] == "user" and "質問:" in msg["content"]:
                if "質問A" in msg["content"]:
                    content = '{"coverage": "Yes", "reason": "ドキュメントに十分な情報があります"}'
                elif "質問B" in msg["content"]:
                    content = '{"coverage": "Partial", "reason": "ドキュメントに部分的な情報があります"}'
                elif "質問C" in msg["content"]:
                    content = '{"coverage": "No", "reason": "ドキュメントに情報がありません"}'
                else:
                    content = '{"coverage": "No", "reason": "不明な質問です"}'
            elif msg["role"] == "user" and "JSON array" in msg["content"]:
                # クエリ生成のモック
                content = '["検索クエリ1", "検索クエリ2", "検索クエリ3", "検索クエリ4"]'

        choices = [MockChoice(MockMessage(content))]
        return MockResponse(choices)

    # モック関数を設定
    openai.chat.completions.create = mock_create
    yield
    # テスト後に元の関数を復元
    openai.chat.completions.create = original_create
