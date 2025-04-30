"""
クエリ生成モジュール。
質問から検索クエリを生成する。
"""
import json
import time
from typing import List, Dict, Any
import yaml
import os

class QueryGenerator:
    """検索クエリを生成するクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定情報（APIキー、モデル名など）
        """
        self.config = config
        self.query_gen_model = config["models"]["query_gen"]
        self.provider = config["provider"]
        
        # APIクライアントの初期化
        if self.provider == "openai":
            import openai
            openai.api_key = config["openai"]["api_key"]
            openai.api_base = config["openai"]["api_base"]
            self.client = openai.OpenAI()
        elif self.provider == "azure_openai":
            import openai
            openai.api_key = config["azure_openai"]["api_key"]
            openai.api_base = config["azure_openai"]["api_base"]
            openai.api_type = "azure"
            openai.api_version = config["azure_openai"]["api_version"]
            self.client = openai.AzureOpenAI()
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config["gemini"]["api_key"])
            self.client = genai
        elif self.provider == "claude":
            import anthropic
            self.client = anthropic.Anthropic(api_key=config["claude"]["api_key"])
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def expand_query(self, question: str, retries: int = 3) -> List[str]:
        """
        質問から4つの多様な検索クエリを生成する
        
        Args:
            question: 元の質問
            retries: リトライ回数
            
        Returns:
            生成された検索クエリのリスト（4つのクエリ）
        """
        system_prompt = "あなたは検索エキスパートです。与えられた質問から、多様な観点でドキュメントを検索するための検索クエリを生成してください。"
        user_prompt = f'質問: "{question}"\nJSON array で 4 つの多様な検索クエリだけを出力してください。'
        
        for attempt in range(retries):
            try:
                if self.provider == "openai" or self.provider == "azure_openai":
                    response = self.client.chat.completions.create(
                        model=self.query_gen_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.5,
                        response_format={"type": "json_object"}
                    )
                    response_text = response.choices[0].message.content
                    
                    # JSON処理
                    try:
                        queries = json.loads(response_text)
                        # JSON配列または"queries"キーを持つオブジェクトからクエリを抽出
                        if isinstance(queries, list):
                            return queries
                        elif isinstance(queries, dict) and "queries" in queries:
                            return queries["queries"]
                        else:
                            # 文字列としてJSONを探す
                            import re
                            json_match = re.search(r'\[.*\]', response_text)
                            if json_match:
                                return json.loads(json_match.group(0))
                            else:
                                # デフォルトとして、オリジナルクエリと一般化したものを返す
                                return [question, question.replace("具体的", ""), 
                                       question.split("、")[0] if "、" in question else question, 
                                       " ".join([w for w in question.split() if len(w) > 1])]
                    except Exception as e:
                        print(f"JSON parsing error: {e}")
                        print(f"Raw response: {response_text}")
                        
                        # 文字列から直接クエリを抽出する試み
                        import re
                        queries = re.findall(r'"([^"]+)"', response_text)
                        if len(queries) >= 4:
                            return queries[:4]
                        else:
                            # デフォルトとして、オリジナルクエリを返す
                            return [question] * 4
                
                elif self.provider == "gemini":
                    response = self.client.generate_content(
                        model=self.query_gen_model,
                        contents=[
                            {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}
                        ]
                    )
                    response_text = response.text
                    
                    # JSON処理（上記と同様）
                    try:
                        # 文字列からJSONを抽出する試み
                        import re
                        json_match = re.search(r'\[.*\]', response_text)
                        if json_match:
                            return json.loads(json_match.group(0))
                        else:
                            # デフォルトとして、オリジナルクエリを返す
                            return [question] * 4
                    except Exception as e:
                        print(f"JSON parsing error: {e}")
                        return [question] * 4
                
                elif self.provider == "claude":
                    response = self.client.messages.create(
                        model=self.query_gen_model,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.5
                    )
                    response_text = response.content[0].text
                    
                    # JSON処理（上記と同様）
                    try:
                        # 文字列からJSONを抽出する試み
                        import re
                        json_match = re.search(r'\[.*\]', response_text)
                        if json_match:
                            return json.loads(json_match.group(0))
                        else:
                            # デフォルトとして、オリジナルクエリを返す
                            return [question] * 4
                    except Exception as e:
                        print(f"JSON parsing error: {e}")
                        return [question] * 4
                
            except Exception as e:
                print(f"Error in query generation (attempt {attempt+1}/{retries}): {e}")
                time.sleep(2 ** attempt)  # 指数バックオフ
        
        # すべてのリトライが失敗した場合は元のクエリを返す
        return [question] * 4


def expand_query(question: str, config_path: str = None) -> List[str]:
    """
    質問から検索クエリを生成する
    
    Args:
        question: 元の質問
        config_path: 設定ファイルのパス（デフォルト: None）
        
    Returns:
        生成された検索クエリのリスト（4つのクエリ）
    """
    # 設定ファイルの読み込み
    if config_path is None:
        config_path = "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # クエリ生成器の初期化
    query_generator = QueryGenerator(config)
    
    # クエリの生成
    queries = query_generator.expand_query(question)
    
    return queries
