# 質問カバー率測定パイプライン

ドキュメントセットと質問リストから、質問がドキュメントでカバーされるか（Yes / Partial / No）を判定し、カバー率を計測するCLIパイプライン。

## 概要

大規模なドキュメントセット（PDF / TXT / DOCX / XLSX等）と多数の質問を対象に、各質問が所与のドキュメントで十分に回答可能かを判定します。
LLMを活用して質問生成、検索、判定を行い、最終的なカバー率を計測します。

## 特徴

- 多様なドキュメント形式対応（拡張可能なExtractor抽象クラス）
- FAISSによる高速ベクトル検索
- 複数のLLMプロバイダ対応（OpenAI / Azure OpenAI / Google Gemini / Anthropic Claude）
- 並列処理による高速な評価

## インストール

```bash
# 開発モードでインストール
pip install -e .
```

## 使用方法

### 1. 設定ファイルの作成

```bash
# 設定ファイルのテンプレートを生成
covercli config
```

生成された`config.yaml`を編集して、APIキーやモデル名を設定します。

### 2. ドキュメントの取り込み

```bash
# docs/以下のドキュメントを取り込み、チャンク化してParquetに保存
covercli ingest docs/ --out chunks.parquet --verbose
```

### 3. インデックスの構築

```bash
# Parquetファイルからベクトルインデックスを構築
covercli build-index chunks.parquet vector_index/ --verbose
```

### 4. 評価の実行

```bash
# 質問リストを読み込み、ドキュメントのカバー率を評価
covercli run-eval questions.tsv vector_index/ results.json \\
    --provider openai \\
    --concurrency 10 \\
    --verbose
```

## 入出力

### 入力

| 種別 | 形式 | 説明 |
|------|------|------|
| ドキュメント | `docs/` 以下に PDF / TXT / DOCX / XLSX 等 | OCR は不要想定。新形式追加時は Extractor 抽象クラスを実装。|
| 質問リスト | TSV または CSV (`questions.tsv`) | 1 行 1 質問。ヘッダ: `id`, `question` |
| 設定ファイル | `config.yaml` | API プロバイダ, モデル名, 同時実行数など |

### 出力

`results.json` には以下の情報が含まれます：

- 取得したドキュメントの抜粋
- 質問ごとの評価結果（カバー率と理由）
- カバー率の集計指標

## 処理フロー

```
1. ingest      → 2. build-index      → 3. run-eval      → results.json
  ↓                  ↓                     ↓
ドキュメント取り込み → ベクトルインデックス構築 → 質問評価 → 結果集計
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 関連情報

- GitHub: https://github.com/yourusername/dbcoverageeval
- バグ報告: https://github.com/yourusername/dbcoverageeval/issues
`,
  `file_path`: `/Users/jiro/development/dbcoverageeval/README.md`
}
レスポンス

{"type":"create","filePath":"/Users/jiro/development/dbcoverageeval/README.md","content":"# 質問カバー率測定パイプライン\n\nドキュメントセットと質問リストから、質問がドキュメントでカバーされるか（Yes / Partial / No）を判定し、カバー率を計測するCLIパイプライン。\n\n## 概要\n\n大規模なドキュメントセット（PDF / TXT / DOCX / XLSX等）と多数の質問を対象に、各質問が所与のドキュメントで十分に回答可能かを判定します。\nLLMを活用して質問生成、検索、判定を行い、最終的なカバー率を計測します。\n\n## 特徴\n\n- 多様なドキュメント形式対応（拡張可能なExtractor抽象クラス）\n- FAISSによる高速ベクトル検索\n- 複数のLLMプロバイダ対応（OpenAI / Azure OpenAI / Google Gemini / Anthropic Claude）\n- 並列処理による高速な評価\n\n## インストール\n\n```bash\n# 開発モードでインストール\npip install -e .\n```\n\n## 使用方法\n\n### 1. 設定ファイルの作成\n\n```bash\n# 設定ファイルのテンプレートを生成\ncovercli config\n```\n\n生成された`config.yaml`を編集して、APIキーやモデル名を設定します。\n\n### 2. ドキュメントの取り込み\n\n```bash\n# docs/以下のドキュメントを取り込み、チャンク化してParquetに保存\ncovercli ingest docs/ --out chunks.parquet --verbose\n```\n\n### 3. インデックスの構築\n\n```bash\n# Parquetファイルからベクトルインデックスを構築\ncovercli build-index chunks.parquet vector_index/ --verbose\n```\n\n### 4. 評価の実行\n\n```bash\n# 質問リストを読み込み、ドキュメントのカバー率を評価\ncovercli run-eval questions.tsv vector_index/ results.json \\\n    --provider openai \\\n    --concurrency 10 \\\n    --verbose\n```\n\n## 入出力\n\n### 入力\n\n| 種別 | 形式 | 説明 |\n|------|------|------|\n| ドキュメント | `docs/` 以下に PDF / TXT / DOCX / XLSX 等 | OCR は不要想定。新形式追加時は Extractor 抽象クラスを実装。|\n| 質問リスト | TSV または CSV (`questions.tsv`) | 1 行 1 質問。ヘッダ: `id`, `question` |\n| 設定ファイル | `config.yaml` | API プロバイダ, モデル名, 同時実行数など |\n\n### 出力\n\n`results.json` には以下の情報が含まれます：\n\n- 取得したドキュメントの抜粋\n- 質問ごとの評価結果（カバー率と理由）\n- カバー率の集計指標\n\n## 処理フロー\n\n```\n1. ingest      → 2. build-index      → 3. run-eval      → results.json\n  ↓                  ↓                     ↓\nドキュメント取り込み → ベクトルインデックス構築 → 質問評価 → 結果集計\n```\n\n## ライセンス\n\nこのプロジェクトはMITライセンスの下で公開されています。\n\n## 関連情報\n\n- GitHub: https://github.com/yourusername/dbcoverageeval\n- バグ報告: https://github.com/yourusername/dbcoverageeval/issues\n