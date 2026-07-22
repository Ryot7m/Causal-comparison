# Causal Inference Platform

FastAPIを用いた因果推論分析プラットフォームです。

CSVをアップロードすると

- セグメンテーション
- AIPW推定
- DR-CDF
- HEI

を自動実行し、
API経由で分析結果を取得できます。

## この分析で分かること

- どの施策が最も効果的か
- どの利用者層に効果があるか
- 平均値だけでは分からない評価分布の変化
- 因果推論による比較可能なセグメント設計

例えば、

「情報量を改善すると推薦意向は向上するのか？」

「期待度が高い利用者ほど更新頻度は重要なのか？」

といった施策効果を観察データから推定できます。


                        Client
                           │
                           │ CSV Upload
                           ▼
                 +------------------+
                 |     FastAPI      |
                 |      api.py      |
                 +------------------+
                           │
                           ▼
                 +------------------+
                 |   services.py    |
                 | Analysis Pipeline|
                 +------------------+
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 segmentation.py      aipw.py          drcdf.py
        │
        └──────────────────┐
                           ▼
                        hei.py
                           │
                           ▼
                    JSON Response

### ATE
ATEを用いて、各情報品質項目が推薦意向に与える平均的な因果効果を推定しました。

![ATE](https://private-user-images.githubusercontent.com/107174339/617125856-8ba8b228-effa-428a-be37-d9757949d605.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODMyNDExMzcsIm5iZiI6MTc4MzI0MDgzNywicGF0aCI6Ii8xMDcxNzQzMzkvNjE3MTI1ODU2LThiYThiMjI4LWVmZmEtNDI4YS1iZTM3LWQ5NzU3OTQ5ZDYwNS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNzA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDcwNVQwODQwMzdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01MWY5MmM2MWZjZjIxMjBmZDQ1ZDg5NDZkY2U4ODgzODQ0ZDY5OTQ0NzBiMjk1MjM1ZDQwNDczYjI1OTJjMThjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.AxK5T_0_c2zqghaL3WGPV5FORoTDfB4qlzRYzeUKmOE)

### DR-CDF

平均値だけでは把握できない変化を確認するため、DR-CDFを用いて推薦意向の分布全体を比較しました。

![DR-CDF正確性](https://github.com/Ryot7m/Causal-comparison/blob/main/png/%E6%AD%A3%E7%A2%BA%E6%80%A7.png)

![DR-CDF更新頻度](https://github.com/Ryot7m/Causal-comparison/blob/main/png/%E6%9B%B4%E6%96%B0%E9%A0%BB%E5%BA%A6.png)

![DR-CDF豊富さ](https://github.com/Ryot7m/Causal-comparison/blob/main/png/%E8%B1%8A%E5%AF%8C%E3%81%95.png)

![DR-CDF詳細さ](https://github.com/Ryot7m/Causal-comparison/blob/main/png/%E8%A9%B3%E7%B4%B0%E3%81%95.png)

### HEI

HEIにより提案手法と比較手法のセグメント品質を比較した

![HEI](https://github.com/Ryot7m/Causal-comparison/blob/main/png/HEI.png)

 ## Directory Structure

```text
causal-inference-platform/
│
├── app/
│   ├── main.py          # FastAPIアプリケーション
│   ├── api.py           # APIエンドポイント
│   ├── services.py      # 分析パイプライン
│   ├── schemas.py       # Pydanticモデル
│   ├── config.py        # 各種設定
│   └── database.py      # DB接続（拡張用）
│
├── workspace/
│   ├── segmentation.py  # セグメンテーション
│   ├── aipw.py          # AIPW推定
│   ├── drcdf.py         # DR-CDF推定
│   ├── hei.py           # HEI算出
│   └── ateplot.py       # 可視化
│
├── tests/               # テストコード
│
├── main.py              # 研究用実行スクリプト
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## API仕様

本システムでは、REST APIを通じて因果推論分析を実行できます。
CSVファイルをアップロードすると、セグメンテーション、AIPWによる平均処置効果推定、DR-CDFによる分布効果推定、HEIによる異質性評価を実施し、JSON形式で結果を返します。

### 因果効果推定API

CSVデータをアップロードし、セグメンテーション・AIPW・DR-CDF・HEIによる分析結果を取得します。

### エンドポイント

**POST**

```
/api/estimate
```

### リクエスト

**Content-Type**

```
multipart/form-data
```

**パラメータ**

| パラメータ | 型 | 説明 |
|------------|----|------|
| file | CSVファイル | 分析対象のデータセット |

### レスポンス

```json
{
  "segment": {
    "cut1": 2.41,
    "cut2": 3.78
  },
  "ate": [
    ...
  ],
  "drcdf": [
    ...
  ],
  "hei": {
    ...
  }
}
```

### レスポンス項目

| 項目 | 説明 |
|------|------|
| segment | 推定されたセグメント境界 |
| ate | セグメントごとの平均処置効果（ATE） |
| drcdf | DR-CDFによる処置効果分布 |
| hei | 異質性評価指標（HEI） |

### エラーレスポンス

| ステータスコード | 内容 |
|-----------------|------|
| 400 | 入力データに必要な列が存在しないなどの入力エラー |
| 500 | サーバ内部で予期しないエラーが発生 |