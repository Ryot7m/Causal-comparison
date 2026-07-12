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

def hei_result(nuis, A0, Y0, S0 ,per_seg_summary):
    # =========================================================
    # pseudo_outcomes の定義（関数の外で再計算）
    # =========================================================
    # 観測されたアウトカムをスコア（0〜4）に変換
    score_arr = np.asarray([0, 1, 2, 3, 4], dtype=float)
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
    score_proposed = calculate_heterogeneity_score(aipw_pseudo_outcomes, -S0)

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

    # HEIスコア（例）
    hei_proposed = [2.374609039743838, 5.471208523395044, 4.0615359970285985, 4.325312698877328]
    hei_rf = [3.1007619303123373, 3.1530651964461756, 2.296957473061643, 2.3988665311251305]

    # 平均値と標準偏差（エラーバー用）の計算
    means = [np.mean(hei_proposed), np.mean(hei_rf)]
    stds = [np.std(hei_proposed, ddof=1), np.std(hei_rf, ddof=1)]

    labels = ['Proposed Method\n(-S0)', 'Causal Clustering\n(Random Forest)']
    x = np.arange(len(labels))

    plt.figure(figsize=(7, 6))

    # 1. 平均値の棒グラフを描画
    bars = plt.bar(x, means, yerr=stds, capsize=10, 
                color=['#ff9999', '#9999ff'], alpha=0.8, edgecolor='black', linewidth=1.2)

    # 2. 各Foldの実際のスコアを「ドット」として重ねて描画（これが論文で信頼される書き方です）
    # ドットが重ならないようにX軸方向に少しだけ散らします（ジッター）
    jitter_proposed = np.random.normal(0, 0.04, size=len(hei_proposed))
    jitter_rf = np.random.normal(1, 0.04, size=len(hei_rf))

    plt.scatter(jitter_proposed, hei_proposed, color='darkred', zorder=3, alpha=0.9, label='Each Fold Score')
    plt.scatter(jitter_rf, hei_rf, color='darkblue', zorder=3, alpha=0.9)

    # 3. 装飾
    plt.ylabel('Normalized Heterogeneity Score (HEI)', fontsize=12)
    plt.title('Comparison of Segmentation Quality (HEI)', fontsize=14)
    plt.xticks(x, labels, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')

    # グラフの余白を自動調整
    plt.tight_layout()
    plt.show()