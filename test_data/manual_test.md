# 質問カバー率測定パイプライン マニュアルテスト手順書

このドキュメントでは、質問カバー率測定パイプラインのマニュアルテスト手順について説明します。提供されたテストデータと質問リストを使用して、パイプラインの各ステップを検証します。

## 1. 準備

### 1.1 テストデータの確認

以下のテストデータが用意されています：

- `/test_data/ai_basics.txt` - 人工知能と機械学習の基礎についてのテキストドキュメント
- `/test_data/data_science.txt` - データサイエンスの基本ガイドについてのテキストドキュメント
- `/test_data/questions.tsv` - 20個のテスト質問リスト

### 1.2 ディレクトリ構造の作成

必要に応じて以下のディレクトリ構造を作成します：

```
/test_data/
  ├── docs/     # テスト用ドキュメントがここに格納されます
  ├── ai_basics.txt
  ├── data_science.txt
  └── questions.tsv
```

テキストファイルを `docs/` ディレクトリにコピーするか、シンボリックリンクを作成します：

```bash
mkdir -p /Users/jiro/development/dbcoverageeval/test_data/docs
cp /Users/jiro/development/dbcoverageeval/test_data/*.txt /Users/jiro/development/dbcoverageeval/test_data/docs/
```

### 1.3 設定ファイルの作成

パイプラインの設定ファイルを作成します：

```bash
# テスト用のconfig.yamlを生成
cd /Users/jiro/development/dbcoverageeval
python -m dbcoverageeval.cli config ./test_data/config.yaml
```

生成された `config.yaml` ファイルを開いて、APIキーを設定します。必要に応じて他の設定も調整します。

## 2. テスト手順

### 2.1 ドキュメント取り込みテスト

ドキュメント取り込み（ingest）の機能をテストします：

```bash
cd /Users/jiro/development/dbcoverageeval
python -m dbcoverageeval.cli ingest ./test_data/docs/ \
    --out ./test_data/chunks.parquet \
    --config ./test_data/config.yaml \
    --verbose
```

#### 検証項目

- コマンドが正常に実行されること
- 出力ファイル `./test_data/chunks.parquet` が生成されること
- 取り込まれたチャンク数、ドキュメント数、ページ数が表示されること

### 2.2 インデックス構築テスト

インデックス構築（build-index）の機能をテストします：

```bash
cd /Users/jiro/development/dbcoverageeval
python -m dbcoverageeval.cli build-index ./test_data/chunks.parquet \
    ./test_data/vector_index/ \
    --config ./test_data/config.yaml \
    --verbose
```

#### 検証項目

- コマンドが正常に実行されること
- 出力ディレクトリ `./test_data/vector_index/` が生成されること
- その中に以下のファイルが生成されること：
  - `vector.index`
  - `meta.parquet`
  - `index_config.pkl`
- インデックス情報（次元数、チャンク数、埋め込みモデル名など）が表示されること

### 2.3 評価テスト

質問カバー率評価（run-eval）の機能をテストします：

```bash
cd /Users/jiro/development/dbcoverageeval
python -m dbcoverageeval.cli run-eval ./test_data/questions.tsv \
    ./test_data/vector_index/ \
    ./test_data/results.json \
    --config ./test_data/config.yaml \
    --concurrency 2 \
    --verbose
```

#### 検証項目

- コマンドが正常に実行されること
- 出力ファイル `./test_data/results.json` が生成されること
- 質問ごとのカバー率（Yes/Partial/No）が表示されること
- カバー率の集計結果（Yes, Partial, No の件数と割合）が表示されること

## 3. 結果の検証

### 3.1 期待される結果

このテストデータと質問セットでは、以下のような結果が期待されます：

- 質問 q_001〜q_010（AIに関する質問）：主に ai_basics.txt 文書から回答可能
- 質問 q_011〜q_019（データサイエンスに関する質問）：主に data_science.txt 文書から回答可能
- 質問 q_020（量子コンピューティングとAIの関係）：両方の文書に明示的な記述がないため「No」または「Partial」

### 3.2 詳細分析

results.json ファイルをテキストエディタで開いて、以下の点を確認します：

1. 各質問に対する判定結果（coverage）が適切か
2. 判定理由（reason）が妥当か
3. 参照されているドキュメント（doc_ids）が関連性の高いものか
4. 抽出されたドキュメント内容（retrieved_docs）が質問に関連するものか

### 3.3 追加検証

さらに詳細な検証として、以下の観点からも確認します：

#### 3.3.1 異なるLLMプロバイダでのテスト

可能であれば、異なるLLMプロバイダ（OpenAI, Claude, Gemini など）で同じテストを実行し、結果を比較します：

```bash
python -m dbcoverageeval.cli run-eval ./test_data/questions.tsv \
    ./test_data/vector_index/ \
    ./test_data/results_claude.json \
    --config ./test_data/config.yaml \
    --provider claude \
    --verbose
```

#### 3.3.2 同時実行数の影響

異なる同時実行数でテストし、速度と結果の違いを確認します：

```bash
# 同時実行数を1に設定
python -m dbcoverageeval.cli run-eval ./test_data/questions.tsv \
    ./test_data/vector_index/ \
    ./test_data/results_conc1.json \
    --config ./test_data/config.yaml \
    --concurrency 1 \
    --verbose

# 同時実行数を10に設定
python -m dbcoverageeval.cli run-eval ./test_data/questions.tsv \
    ./test_data/vector_index/ \
    ./test_data/results_conc10.json \
    --config ./test_data/config.yaml \
    --concurrency 10 \
    --verbose
```

#### 3.3.3 パラメータ感度分析

異なるパラメータ設定（チャンクサイズ、オーバーラップ、トップK値など）でテストし、結果の違いを確認します。

## 4. バグ・性能問題の報告方法

テスト中にバグや性能問題が見つかった場合は、以下の情報を含めてGitHub Issueを作成してください：

1. 問題の詳細な説明
2. 再現手順
3. 使用した設定（config.yamlの内容）
4. エラーメッセージやスタックトレース（該当する場合）
5. 期待される動作と実際の動作の違い
6. 使用した環境（OSバージョン、Pythonバージョン、主要ライブラリのバージョン）

## 5. パフォーマンス測定

実行時間とリソース使用状況を測定するには、以下のコマンドを使用します：

```bash
# 時間測定
time python -m dbcoverageeval.cli run-eval ./test_data/questions.tsv \
    ./test_data/vector_index/ \
    ./test_data/results.json \
    --config ./test_data/config.yaml

# メモリ使用量測定（macOS/Linux）
/usr/bin/time -l python -m dbcoverageeval.cli run-eval ./test_data/questions.tsv \
    ./test_data/vector_index/ \
    ./test_data/results.json \
    --config ./test_data/config.yaml
```

特に以下の点に注目してパフォーマンスを評価します：

1. インデックス構築の時間と使用メモリ
2. 質問評価の平均処理時間
3. 同時実行数を増やした場合のスケーラビリティ
4. APIコール数と使用トークン数（コスト）

## まとめ

このマニュアルテスト手順書に従って、質問カバー率測定パイプラインの機能と性能を検証してください。テスト結果は、品質保証のための貴重な情報源となります。問題や改善点が見つかった場合は、速やかに報告してください。
