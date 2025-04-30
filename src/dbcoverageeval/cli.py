"""
CLIインターフェース。
各コマンド（ingest, build-index, run-eval）を実行するためのインターフェース。
"""
import os
import sys
import typer
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import yaml
import concurrent.futures
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
import time
import json

from . import ingest as ingest_module
from . import index as index_module
from . import querygen as querygen_module
from . import search as search_module
from . import judge as judge_module
from . import report as report_module

# Typer アプリの作成
app = typer.Typer(
    name="covercli",
    help="質問カバー率測定パイプライン",
    add_completion=False
)

console = Console()

@app.command()
def ingest(
    docs_path: str = typer.Argument(..., help="ドキュメントが格納されているディレクトリのパス"),
    out_parquet: str = typer.Option("chunks.parquet", "--out", "-o", help="出力するParquetファイルのパス"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="設定ファイルのパス"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="詳細な出力を表示する")
):
    """
    ドキュメントを取り込み、チャンク化してParquetに保存する
    """
    console.print(f"[bold green]ドキュメント取り込み処理を開始します: {docs_path}[/bold green]")
    
    # 設定ファイルの読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    chunk_tokens = config.get("chunk_tokens", 1000)
    chunk_overlap = config.get("chunk_overlap", 200)
    
    console.print(f"チャンクサイズ: {chunk_tokens} トークン, オーバーラップ: {chunk_overlap} トークン")
    
    # 取り込み処理
    start_time = time.time()
    ingest_module.ingest(docs_path, out_parquet, chunk_tokens, chunk_overlap)
    elapsed = time.time() - start_time
    
    # 結果の確認
    if os.path.exists(out_parquet):
        df = pd.read_parquet(out_parquet)
        console.print(f"[bold green]取り込み完了: {len(df)} チャンクを {out_parquet} に保存しました（処理時間: {elapsed:.1f}秒）[/bold green]")
        
        # 詳細情報の表示
        if verbose:
            # ドキュメント数とページ数のカウント
            unique_docs = df["doc_id"].nunique()
            unique_pages = df["parent_id"].nunique()
            
            # テーブルの作成
            table = Table(title="取り込み結果")
            table.add_column("項目", style="cyan")
            table.add_column("値", style="green")
            
            table.add_row("ドキュメント数", str(unique_docs))
            table.add_row("ページ数", str(unique_pages))
            table.add_row("チャンク数", str(len(df)))
            table.add_row("平均チャンク長", f"{df['text'].str.len().mean():.1f} 文字")
            table.add_row("最大チャンク長", f"{df['text'].str.len().max()} 文字")
            
            console.print(table)
    else:
        console.print(f"[bold red]エラー: Parquetファイルが生成されませんでした[/bold red]")
        sys.exit(1)


@app.command("build-index")
def build_index(
    parquet_path: str = typer.Argument(..., help="入力Parquetファイルのパス"),
    out_index_dir: str = typer.Argument(..., help="出力インデックスディレクトリのパス"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="設定ファイルのパス"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="詳細な出力を表示する")
):
    """
    Parquetファイルからベクトルインデックスを構築する
    """
    console.print(f"[bold green]インデックス構築処理を開始します: {parquet_path}[/bold green]")
    
    # 設定ファイルの存在確認
    if not os.path.exists(config_path):
        console.print(f"[bold red]エラー: 設定ファイルが見つかりません: {config_path}[/bold red]")
        sys.exit(1)
    
    # Parquetファイルの存在確認
    if not os.path.exists(parquet_path):
        console.print(f"[bold red]エラー: Parquetファイルが見つかりません: {parquet_path}[/bold red]")
        sys.exit(1)
    
    # 出力ディレクトリの準備
    out_index_dir = Path(out_index_dir)
    out_index_dir.mkdir(parents=True, exist_ok=True)
    
    # インデックス構築
    start_time = time.time()
    index_module.build_index(parquet_path, out_index_dir, config_path)
    elapsed = time.time() - start_time
    
    # 結果の確認
    if os.path.exists(out_index_dir / "vector.index"):
        console.print(f"[bold green]インデックス構築完了: {out_index_dir} に保存しました（処理時間: {elapsed:.1f}秒）[/bold green]")
        
        # 詳細情報の表示
        if verbose:
            # インデックス情報の読み込み
            import pickle
            with open(out_index_dir / "index_config.pkl", 'rb') as f:
                index_config = pickle.load(f)
            
            # メタデータの読み込み
            meta_df = pd.read_parquet(out_index_dir / "meta.parquet")
            
            # テーブルの作成
            table = Table(title="インデックス情報")
            table.add_column("項目", style="cyan")
            table.add_column("値", style="green")
            
            table.add_row("次元数", str(index_config["dimension"]))
            table.add_row("チャンク数", str(index_config["num_chunks"]))
            table.add_row("埋め込みモデル", index_config["embedding_model"])
            table.add_row("親ドキュメント数", str(meta_df["parent_id"].nunique()))
            
            console.print(table)
    else:
        console.print(f"[bold red]エラー: インデックスファイルが生成されませんでした[/bold red]")
        sys.exit(1)


def _load_questions(questions_path: str) -> List[Dict[str, Any]]:
    """
    質問リストを読み込む
    
    Args:
        questions_path: 質問リストのファイルパス（TSVまたはCSV）
        
    Returns:
        質問のリスト（id, questionを含む）
    """
    # ファイル拡張子からデリミタを推測
    ext = os.path.splitext(questions_path)[1].lower()
    delimiter = '\t' if ext == '.tsv' else ','
    
    # 質問リストの読み込み
    df = pd.read_csv(questions_path, delimiter=delimiter)
    
    # 必須カラムの確認
    required_columns = ["id", "question"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"質問リストに必須カラムがありません: {', '.join(missing_columns)}")
    
    # 質問リストの変換
    questions = df[["id", "question"]].to_dict('records')
    
    return questions


def _process_question(
    q_id: str, 
    question: str, 
    index_dir: str, 
    config_path: str, 
    config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    1つの質問を処理する
    
    Args:
        q_id: 質問ID
        question: 質問文
        index_dir: インデックスディレクトリのパス
        config_path: 設定ファイルのパス
        config: 設定情報
        
    Returns:
        質問ID、判定結果、親ドキュメント情報のリストのタプル
    """
    try:
        # クエリ生成
        queries = querygen_module.expand_query(question, config_path)
        
        # 検索
        top_k_child = config.get("top_k_child", 25)
        parent_results = search_module.retrieve(queries, index_dir, config_path, k=top_k_child)
        
        # チャンク情報の読み込み
        chunks_df = pd.read_parquet(Path(index_dir) / "meta.parquet")
        chunks_df_with_text = pd.read_parquet(config.get("chunks_path", "chunks.parquet"))
        merged_df = pd.merge(chunks_df, chunks_df_with_text[["chunk_id", "text"]], on="chunk_id")
        
        # 判定
        judgment = judge_module.judge(question, parent_results, merged_df, config_path)
        
        return q_id, judgment, parent_results
    
    except Exception as e:
        # エラーが発生した場合は、エラーメッセージを含むデフォルト値を返す
        default_judgment = {
            "coverage": "No",
            "reason": f"エラー: {str(e)}"
        }
        return q_id, default_judgment, []


@app.command("run-eval")
def run_eval(
    questions_path: str = typer.Argument(..., help="質問リストのファイルパス（TSVまたはCSV）"),
    index_dir: str = typer.Argument(..., help="インデックスディレクトリのパス"),
    out_json: str = typer.Argument(..., help="出力するJSONファイルのパス"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="設定ファイルのパス"),
    concurrency: int = typer.Option(None, "--concurrency", "-n", help="同時実行数"),
    provider: str = typer.Option(None, "--provider", "-p", help="APIプロバイダ（config.yamlの設定を上書き）"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="詳細な出力を表示する"),
    top_parent: int = typer.Option(None, "--top_parent", help="取得する親ドキュメントの最大数")
):
    """
    質問とドキュメントからカバー率を評価する
    """
    console.print(f"[bold green]評価処理を開始します[/bold green]")
    
    # 設定ファイルの読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # コマンドラインオプションによる設定の上書き
    if provider:
        config["provider"] = provider
    
    if concurrency is None:
        concurrency = config.get("concurrency", 1)
    
    if top_parent is not None:
        config["top_k_parent"] = top_parent
    
    # 質問リストの読み込み
    questions = _load_questions(questions_path)
    console.print(f"質問リスト読み込み完了: {len(questions)} 件")
    
    # チャンクパスの設定
    if "chunks_path" not in config:
        # デフォルトとして、インデックスディレクトリの親ディレクトリにあるchunks.parquetを参照
        config["chunks_path"] = os.path.join(os.path.dirname(os.path.dirname(index_dir)), "chunks.parquet")
    
    # 並列処理のセットアップ
    results = {}
    parent_results_map = {}
    
    # 進捗表示の設定
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("残り"),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]質問処理中...", total=len(questions))
        
        # 並列処理
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # タスクの送信
            futures = {
                executor.submit(
                    _process_question, 
                    q["id"], 
                    q["question"], 
                    index_dir, 
                    config_path, 
                    config
                ): q["id"] for q in questions
            }
            
            # 結果の収集
            for future in concurrent.futures.as_completed(futures):
                q_id = futures[future]
                try:
                    q_id, judgment, parent_results = future.result()
                    results[q_id] = judgment
                    parent_results_map[q_id] = parent_results
                    
                    if verbose:
                        coverage = judgment["coverage"]
                        color = {
                            "Yes": "green",
                            "Partial": "yellow",
                            "No": "red"
                        }.get(coverage, "white")
                        console.print(f"[{color}]質問 {q_id}: {coverage}[/{color}] - {judgment['reason']}")
                
                except Exception as e:
                    console.print(f"[bold red]エラー（質問 {q_id}）: {str(e)}[/bold red]")
                    results[q_id] = {
                        "coverage": "No",
                        "reason": f"処理エラー: {str(e)}"
                    }
                
                # 進捗を更新
                progress.update(task_id, advance=1)
    
    # チャンク情報の読み込み
    chunks_df = pd.read_parquet(config["chunks_path"])
    
    # 結果の保存
    report_module.save(out_json, questions, parent_results_map, chunks_df, results)
    
    # 結果の表示
    with open(out_json, 'r', encoding='utf-8') as f:
        json_results = json.load(f)
    
    metrics = json_results["metrics"]
    console.print("\n[bold green]評価結果[/bold green]")
    
    # テーブルの作成
    table = Table(title="カバレッジ集計")
    table.add_column("カバー率", style="cyan")
    table.add_column("件数", style="green")
    table.add_column("割合", style="yellow")
    
    table.add_row("Yes", str(metrics["yes"]), f"{metrics['coverage_rate_yes']:.1%}")
    table.add_row("Partial", str(metrics["partial"]), "-")
    table.add_row("No", str(metrics["no"]), "-")
    table.add_row("Yes or Partial", str(metrics["yes"] + metrics["partial"]), f"{metrics['coverage_rate_yes_or_partial']:.1%}")
    
    console.print(table)
    console.print(f"\n詳細な結果は {out_json} に保存されました。")


@app.command("config")
def generate_config(
    out_path: str = typer.Argument("config.yaml", help="出力する設定ファイルのパス")
):
    """
    設定ファイルのテンプレートを生成する
    """
    config_template = """provider: openai  # azure_openai / gemini / claude も可
models:
  embedding: text-embedding-3-small
  query_gen: gpt-4o-mini
  judge: gpt-4o
openai:
  api_key: ${OPENAI_API_KEY}
  api_base: https://api.openai.com/v1
azure_openai:
  api_key: ${AZURE_OPENAI_API_KEY}
  api_base: https://your-resource.openai.azure.com/
  api_version: 2023-05-15
gemini:
  api_key: ${GEMINI_API_KEY}
claude:
  api_key: ${CLAUDE_API_KEY}
concurrency: 10
chunk_tokens: 1000
chunk_overlap: 200
top_k_child: 25
top_k_parent: 100
max_parents_for_judge: 3
"""
    
    # ファイルへの書き込み
    with open(out_path, 'w') as f:
        f.write(config_template)
    
    console.print(f"[bold green]設定ファイルのテンプレートを {out_path} に生成しました[/bold green]")
    console.print("環境変数を設定するか、APIキーを直接入力してください。")


if __name__ == "__main__":
    app()
