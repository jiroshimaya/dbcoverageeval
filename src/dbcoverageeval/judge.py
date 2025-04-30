"""
判定モジュール。
質問とドキュメントから、カバー率を判定する。
"""
import json
import time
import yaml
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

class Judge:
    """質問とドキュメントからカバー率を判定するクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定情報（APIキー、モデル名など）
        """
        self.config = config
        self.judge_model = config["models"]["judge"]
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
    
    def _prepare_excerpts(self, chunks_df: pd.DataFrame, parent_ids: List[str]) -> Dict[str, str]:
        """
        親ドキュメントの抜粋を作成する
        
        Args:
            chunks_df: チャンクのDataFrame
            parent_ids: 親ドキュメントIDのリスト
            
        Returns:
            親ドキュメントIDをキー、抜粋を値とする辞書
        """
        excerpts = {}
        
        for parent_id in parent_ids:
            # 親に所属するチャンクを取得
            parent_chunks = chunks_df[chunks_df["parent_id"] == parent_id]
            
            if len(parent_chunks) == 0:
                continue
            
            # 先頭と末尾のチャンクを取得
            first_chunk = parent_chunks.iloc[0]["text"]
            last_chunk = parent_chunks.iloc[-1]["text"] if len(parent_chunks) > 1 else ""
            
            # 抜粋を作成（先頭300文字+末尾300文字）
            first_excerpt = first_chunk[:300]
            last_excerpt = last_chunk[-300:] if last_chunk else ""
            
            # 親ドキュメントIDと抜粋を格納
            if last_excerpt:
                excerpt = f"【抜粋先頭】\n{first_excerpt}\n\n【抜粋末尾】\n{last_excerpt}"
            else:
                excerpt = f"【抜粋】\n{first_excerpt}"
            
            excerpts[parent_id] = excerpt
        
        return excerpts
    
    def judge(self, question: str, parent_results: List[Dict[str, Any]], chunks_df: pd.DataFrame, max_parents: int = 3, retries: int = 3) -> Dict[str, Any]:
        """
        質問とドキュメントから、カバー率を判定する
        
        Args:
            question: 質問
            parent_results: 親ドキュメント情報のリスト
            chunks_df: チャンクのDataFrame
            max_parents: 判定に使用する親ドキュメント数
            retries: リトライ回数
            
        Returns:
            判定結果（coverage, reason）
        """
        # 使用する親ドキュメントを選択（スコア上位）
        selected_parents = parent_results[:max_parents]
        parent_ids = [parent["parent_id"] for parent in selected_parents]
        
        # 親ドキュメントの抜粋を作成
        excerpts = self._prepare_excerpts(chunks_df, parent_ids)
        
        # ドキュメント抜粋テキストを構築
        docs_text = ""
        for i, parent_id in enumerate(parent_ids):
            if parent_id in excerpts:
                docs_text += f"ドキュメント {i+1}: {parent_id}\n{excerpts[parent_id]}\n\n"
        
        # システムプロンプトとユーザープロンプトを構築
        system_prompt = "あなたは厳密なレビュワーです。ドキュメント内に質問に対する十分な回答が含まれているかを評価してください。"
        user_prompt = f"""質問: "{question}"
検索結果:
{docs_text}

以下の strict JSON を出力:
{{"coverage": "Yes|Partial|No", "reason": "..."}}
"""
        
        for attempt in range(retries):
            try:
                if self.provider == "openai" or self.provider == "azure_openai":
                    response = self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    response_text = response.choices[0].message.content
                    
                elif self.provider == "gemini":
                    response = self.client.generate_content(
                        model=self.judge_model,
                        contents=[
                            {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}
                        ],
                        generation_config={"temperature": 0.1}
                    )
                    response_text = response.text
                    
                elif self.provider == "claude":
                    response = self.client.messages.create(
                        model=self.judge_model,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1
                    )
                    response_text = response.content[0].text
                
                # JSON解析
                try:
                    result = json.loads(response_text)
                    
                    # フォーマット検証
                    if not isinstance(result, dict):
                        raise ValueError("Result must be a dictionary")
                    
                    if "coverage" not in result or "reason" not in result:
                        raise ValueError("Result must contain 'coverage' and 'reason' keys")
                    
                    if result["coverage"] not in ["Yes", "Partial", "No"]:
                        raise ValueError("Coverage must be one of 'Yes', 'Partial', 'No'")
                    
                    return result
                    
                except Exception as e:
                    print(f"JSON parsing error (attempt {attempt+1}/{retries}): {e}")
                    print(f"Raw response: {response_text}")
                    
                    # 文字列からcoverageとreasonを抽出する試み
                    import re
                    coverage_match = re.search(r'"coverage"\s*:\s*"(Yes|Partial|No)"', response_text)
                    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', response_text)
                    
                    if coverage_match and reason_match:
                        return {
                            "coverage": coverage_match.group(1),
                            "reason": reason_match.group(1)
                        }
                    
                    # それでも失敗した場合、デフォルト値を返す
                    if attempt == retries - 1:
                        return {
                            "coverage": "No", 
                            "reason": "自動判定に失敗しました。"
                        }
                
            except Exception as e:
                print(f"Error in judgment (attempt {attempt+1}/{retries}): {e}")
                time.sleep(2 ** attempt)  # 指数バックオフ
        
        # すべてのリトライが失敗した場合はデフォルト値を返す
        return {
            "coverage": "No", 
            "reason": "自動判定に失敗しました。リトライ制限に達しました。"
        }


def judge(question: str, parent_results: List[Dict[str, Any]], chunks_df: pd.DataFrame, config_path: str = None) -> Dict[str, Any]:
    """
    質問とドキュメントから、カバー率を判定する
    
    Args:
        question: 質問
        parent_results: 親ドキュメント情報のリスト
        chunks_df: チャンクのDataFrame
        config_path: 設定ファイルのパス（デフォルト: None）
        
    Returns:
        判定結果（coverage, reason）
    """
    # 設定ファイルの読み込み
    if config_path is None:
        config_path = "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    max_parents = config.get("max_parents_for_judge", 3)
    
    # 判定器の初期化
    judge_obj = Judge(config)
    
    # 判定の実行
    result = judge_obj.judge(question, parent_results, chunks_df, max_parents=max_parents)
    
    return result
