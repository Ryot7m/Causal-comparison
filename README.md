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
        │
        ▼
      hei.py
        │
        ▼
   JSON Response

 ## Directory Structure

```text
causal-inference-platform/
│
├── app/
│   ├── main.py          # FastAPIアプリケーション
│   ├── api.py           # APIエンドポイント
│   ├── analysis.py      # 分析パイプライン
│   ├── dantic.py       # Pydanticモデル
│   ├── config.py        # 各種設定
│   └── database.py      # DB接続（拡張用）
│
├── workspace/
│   ├── segmentation.py  # セグメンテーション
    ├── calibrated.py    # 確率較正の共通処理
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

## 推定フロー

本システムでは、CSVデータを入力として受け取り、前処理から因果効果推定までを一連のパイプラインとして実行します。

```text
CSVデータ
    │
    ▼
データ読み込み・入力検証
(load_csv / validate_data)
    │
    ▼
前処理
(preprocess)
    │
    ├── 説明変数の抽出
    ├── One-Hot Encoding
    ├── 処置変数(A)作成
    └── アウトカム(Y)作成
    │
    ▼
セグメンテーション
(segmentation.py)
    │
    ▼
AIPWによる平均処置効果(ATE)推定
(aipw.py)
    │
    ▼
DR-CDFによる分布効果推定
(drcdf.py)
    │
    ▼
HEIによる異質性評価
(hei.py)
    │
    ▼
JSONレスポンス生成
(create_response)
```

### 各処理の概要

| 処理 | 内容 |
|------|------|
| データ読み込み | アップロードされたCSVファイルを読み込む |
| 入力検証 | 必須列の存在を確認する |
| 前処理 | 学習用データ(X・A・Y)を作成する |
| セグメンテーション | セグメント境界を推定する |
| AIPW | 平均処置効果（ATE）を推定する |
| DR-CDF | 処置効果の分布を推定する |
| HEI | セグメント間の異質性を評価する |
| レスポンス生成 | APIレスポンス(JSON)を生成する |

## 技術スタック

| 分類 | 使用技術 |
|------|----------|
| 言語 | Python 3 |
| Webフレームワーク | FastAPI |
| データ処理 | Pandas, NumPy |
| 機械学習 | scikit-learn |
| 因果推論 | AIPW, DR-CDF |
| 可視化 | Matplotlib |
| データ検証 | Pydantic |
| APIドキュメント | Swagger UI（OpenAPI） |
| コンテナ | Docker |
| バージョン管理 | Git / GitHub |

### 使用技術の役割

- **FastAPI**：CSVアップロードおよび分析APIの提供
- **Pandas / NumPy**：データ前処理
- **scikit-learn**：特徴量変換・学習処理
- **AIPW / DR-CDF**：因果効果・分布効果の推定
- **Docker**：実行環境のコンテナ化
- **GitHub Actions**：テストの自動実行（予定）

## 研究アルゴリズム

本プロジェクトでは、解釈性を重視した因果セグメンテーション手法を実装しています。
分析は以下の流れで実行されます。

CSV
 ↓
Segmentation
 ↓
Cross-fitting
 ↓
AIPW
 ↓
DR-CDF
 ↓
HEI

詳細の説明は以下の通りです。

1. **セグメンテーション**
   - 説明変数を基に対象をセグメントへ分類
   - セグメント境界を最適化

2. **AIPW（Augmented Inverse Probability Weighting）**
   - セグメントごとの平均処置効果（ATE）を推定
   - 二重ロバスト性を持つ推定手法

3. **DR-CDF**
   - 処置効果の分布を推定
   - 平均だけでなく分布全体を評価

4. **HEI（Heterogeneity Evaluation Index）**
   - セグメント間の異質性を定量評価
   - 提案手法と比較手法の性能比較に利用

## 評価結果

提案手法を既存の因果クラスタリング手法と比較しました。

評価指標

- Average Treatment Effect (ATE)
- DR-CDF
- Heterogeneity Evaluation Index (HEI)
- Standardized Mean Difference (SMD)
- Effective Sample Size (ESS)

### 比較結果

| 手法 | HEI | 特徴 |
|------|----:|------|
| 提案手法 | ○○ | 高い異質性と解釈性 |
| Causal Clustering | ○○ | 高い異質性 |
| Linear Model | ○○ | 異質性は小さい |

### 分析例

- ATE推定結果
![ATE](https://private-user-images.githubusercontent.com/107174339/617125856-8ba8b228-effa-428a-be37-d9757949d605.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODMyNDExMzcsIm5iZiI6MTc4MzI0MDgzNywicGF0aCI6Ii8xMDcxNzQzMzkvNjE3MTI1ODU2LThiYThiMjI4LWVmZmEtNDI4YS1iZTM3LWQ5NzU3OTQ5ZDYwNS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNzA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDcwNVQwODQwMzdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01MWY5MmM2MWZjZjIxMjBmZDQ1ZDg5NDZkY2U4ODgzODQ0ZDY5OTQ0NzBiMjk1MjM1ZDQwNDczYjI1OTJjMThjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.AxK5T_0_c2zqghaL3WGPV5FORoTDfB4qlzRYzeUKmOE)

- DR-CDF

![DR-CDF正確性](https://github.com/Ryot7m/Causal-comparison/blob/main/png/%E6%AD%A3%E7%A2%BA%E6%80%A7.png)

![DR-CDF更新頻度](https://github.com/Ryot7m/Causal-comparison/blob/main/png/%E6%9B%B4%E6%96%B0%E9%A0%BB%E5%BA%A6.png)

![DR-CDF豊富さ](https://github.com/Ryot7m/Causal-comparison/blob/main/png/%E8%B1%8A%E5%AF%8C%E3%81%95.png)

![DR-CDF詳細さ](https://github.com/Ryot7m/Causal-comparison/blob/main/png/%E8%A9%B3%E7%B4%B0%E3%81%95.png)

- HEI

![HEI](https://github.com/Ryot7m/Causal-comparison/blob/main/png/HEI.png)

- セグメント境界

##### 正確性
| Segment | Treated ($n_1$) | Control ($n_0$) | Max SMD ↓ | Extreme PS Rate ↓ | ESS Ratio ↑ |
|:-------:|----------------:|----------------:|----------:|------------------:|------------:|
| 0 | 39 | 502 | 0.276 | 0.218 | 0.683 |
| 1 | 84 | 257 | 0.377 | 0.012 | 0.228 |
| 2 | 285 | 158 | 0.086 | 0.002 | 0.744 |

##### 更新頻度
| Segment | Treated ($n_1$) | Control ($n_0$) | Max SMD ↓ | Extreme PS Rate ↓ | ESS Ratio ↑ |
|:-------:|----------------:|----------------:|----------:|------------------:|------------:|
| 0 | 51 | 490 | 0.129 | 0.100 | 0.784 |
| 1 | 101 | 240 | 0.065 | 0.000 | 0.937 |
| 2 | 307 | 136 | 0.110 | 0.002 | 0.870 |

##### 豊富さ
| Segment | Treated ($n_1$) | Control ($n_0$) | Max SMD ↓ | Extreme PS Rate ↓ | ESS Ratio ↑ |
|:-------:|----------------:|----------------:|----------:|------------------:|------------:|
| 0 | 62 | 479 | 0.148 | 0.067 | 0.693 |
| 1 | 134 | 207 | 0.053 | 0.000 | 0.923 |
| 2 | 349 | 94 | 0.129 | 0.000 | 0.954 |

##### 詳細さ
| Segment | Treated ($n_1$) | Control ($n_0$) | Max SMD ↓ | Extreme PS Rate ↓ | ESS Ratio ↑ |
|:-------:|----------------:|----------------:|----------:|------------------:|------------:|
| 0 | 56 | 485 | 0.131 | 0.081 | 0.800 |
| 1 | 108 | 233 | 0.072 | 0.000 | 0.948 |
| 2 | 332 | 111 | 0.097 | 0.007 | 0.902 |