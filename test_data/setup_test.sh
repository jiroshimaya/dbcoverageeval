#!/bin/bash
# マニュアルテスト用データをセットアップするスクリプト

# 基本ディレクトリの設定
BASE_DIR=/Users/jiro/development/dbcoverageeval
TEST_DATA_DIR=$BASE_DIR/test_data
DOCS_DIR=$TEST_DATA_DIR/docs

# ディレクトリの作成
mkdir -p $DOCS_DIR

# テキストファイルをdocsディレクトリにコピー
cp $TEST_DATA_DIR/ai_basics.txt $DOCS_DIR/
cp $TEST_DATA_DIR/data_science.txt $DOCS_DIR/

# 設定ファイルの生成（既に存在しない場合）
if [ ! -f "$TEST_DATA_DIR/config.yaml" ]; then
    echo "設定ファイルを生成します..."
    cd $BASE_DIR
    python -m dbcoverageeval.cli config $TEST_DATA_DIR/config.yaml
    echo "設定ファイルが生成されました：$TEST_DATA_DIR/config.yaml"
    echo "このファイルを編集して、APIキーなどを設定してください。"
else
    echo "設定ファイルが既に存在します：$TEST_DATA_DIR/config.yaml"
fi

echo "=== セットアップ完了 ==="
echo "以下のコマンドでテストを開始できます："
echo ""
echo "# ドキュメント取り込み"
echo "cd $BASE_DIR"
echo "python -m dbcoverageeval.cli ingest $DOCS_DIR --out $TEST_DATA_DIR/chunks.parquet --config $TEST_DATA_DIR/config.yaml --verbose"
echo ""
echo "# インデックス構築"
echo "python -m dbcoverageeval.cli build-index $TEST_DATA_DIR/chunks.parquet $TEST_DATA_DIR/vector_index/ --config $TEST_DATA_DIR/config.yaml --verbose"
echo ""
echo "# 評価の実行"
echo "python -m dbcoverageeval.cli run-eval $TEST_DATA_DIR/questions.tsv $TEST_DATA_DIR/vector_index/ $TEST_DATA_DIR/results.json --config $TEST_DATA_DIR/config.yaml --concurrency 2 --verbose"
echo ""
echo "詳細な手順は以下のドキュメントを参照してください：$TEST_DATA_DIR/manual_test.md"
