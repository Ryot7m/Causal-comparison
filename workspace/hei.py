import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_heterogeneity_score(pseudo_outcomes, predicted_scores):
    """
    ユーザーを予測スコアで10等分（デシル）し、
    セグメント間ATEの【不偏分散】をHeterogeneity Scoreとして計算する関数
    """
    # 1. スコアが高い順（降順）に並べ替え
    order = np.argsort(predicted_scores)[::-1]
    sorted_outcomes = pseudo_outcomes[order]
    
    # 2. 10等分に分割
    deciles = np.array_split(sorted_outcomes, 10)
    
    # 3. 各デシル（グループ）の平均因果効果（ATE）を計算
    ates = [np.mean(d) for d in deciles]
    
    # 4. デシル間ATEの【不偏分散（Variance）】を計算
    # ddof=1 とすることで学術統計で一般的な「不偏分散」になります
    heterogeneity_score = np.var(ates, ddof=1)
    
    return heterogeneity_score

def calculate_normalized_hei(pseudo_outcomes, predicted_scores, per_seg_summary):
    heterogeneity = calculate_heterogeneity_score(pseudo_outcomes, predicted_scores)

    smd_penalty = (
        sum(row["n"] * row["seg_mean_abs_SMD"] for row in per_seg_summary)
        / sum(row["n"] for row in per_seg_summary)
    )

    return heterogeneity / smd_penalty

def hei_result(nuis, A0, Y0, S0 , per_seg_summary, score_arr):
    # =========================================================
    # pseudo_outcomes の定義（関数の外で再計算）
    # =========================================================
    # 観測されたアウトカムをスコア（0〜4）に変換
    Z0 = score_arr[Y0]

    # nuis(辞書)から予測値を取り出し、極端な傾向スコアをクリップ
    e0 = np.clip(nuis["e_hat"], 1e-4, 1 - 1e-4)
    mu1_0 = nuis["mu1_hat"]
    mu0_0 = nuis["mu0_hat"]

    # AIPWスコア（個人の擬似アウトカム）を1次元配列として計算
    aipw_pseudo_outcomes = (mu1_0 - mu0_0) + (A0 / e0) * (Z0 - mu1_0) - ((1 - A0) / (1 - e0)) * (Z0 - mu0_0)

    # =========================================================
    # 各モデルのHeterogeneity Scoreを算出
    # =========================================================
    score_proposed = hei_proposed = calculate_normalized_hei(aipw_pseudo_outcomes, -S0, per_seg_summary)

    print(score_proposed)

    # 論文の本文や表（Table）にそのまま載せられるようにDataFrame化
    heterogeneity_results = pd.DataFrame({
        "Method": [
            "Proposed Method (-S0)", 
            # "Causal Clustering (RF)", 
            # "Linear Model (Ridge)"
        ],
        "Heterogeneity Score (ATE Variance)": [
            score_proposed, 
            # score_cluster, 
            # score_linear
        ]
    })

    print("==== Heterogeneity Score (Evaluation of Segmentation Quality) ====")
    print(heterogeneity_results.to_string(index=False))
    
    smd_num = (
    sum(row["n"] * row["seg_mean_abs_SMD"] for row in per_seg_summary)
    /sum(row["n"] for row in per_seg_summary)
    )
    score_proposed /= smd_num
    print(score_proposed)