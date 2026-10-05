"""
アプリケーションに対応させた列目に変更する処理（自分で設定しているカラムは手動で変更も可能）
"""

import json
from pathlib import Path
import pandas as pd

INPUT_CSV = Path("research_sample.csv")
OUTPUT_CSV = Path("causal_sample.csv")
CONFIG_JSON = Path("examples/api_config.json")

df = pd.read_csv(INPUT_CSV, encoding="utf-8")
config = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))

# アプリが必要とする必須列を抽出
treatment = config["treatment"]
treatment_column = treatment["column"] if treatment["mode"] == "binary_column" else treatment["source_column"]
required_columns = [
    treatment_column,
    config["outcome"]["column"],
    config["segment"]["column"],
    *config["covariates"]["columns"],
]

# 既に要件を満たしている列名を除外
needed_columns = [col for col in required_columns if col not in df.columns]

if needed_columns:
    print("\n【手動マッピングが必要です】")
    print(f"現在のCSVのカラム: {list(df.columns)}")
    print(f"不足している必須カラム: {needed_columns}")
    print("-" * 40)
    
    column_map = {}
    for target_col in needed_columns:
        # ユーザーにどの元のカラムを使うか入力させる
        while True:
            source_col = input(f"'{target_col}' に割り当てる元のカラム名を入力してください: ")
            if source_col in df.columns:
                column_map[source_col] = target_col
                break
            else:
                print("エラー: 入力されたカラム名が現在のCSVに存在しません。再度入力してください。")
    
    # ユーザーの入力に基づいて列名を変更
    df = df.rename(columns=column_map)

# 最終確認
missing_target = set(required_columns) - set(df.columns)
if missing_target:
    raise KeyError(f"変換後CSVにアプリ必須列がありません: {sorted(missing_target)}")

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\n保存しました: {OUTPUT_CSV}")