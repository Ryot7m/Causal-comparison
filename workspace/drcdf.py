import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def oc_dr_cdf_by_seg(A, Y, nuis, seg, cap=None, level_labels=None):
    """
    クラス別 DR-CDF（F1-F0）を計算して DataFrame で返す。
    - A: (n,) treatment 0/1
    - Y: (n,) outcome を 0..K-1 にエンコードしたもの（あなたのYと同じ）
    - nuis: fit_cf_oc の戻り（e_hat, p_hat1, p_hat0 が必要）
    - seg: (n,) クラスID（0,1,2...）
    - cap: IF-guard（例: if_g=100）。Noneならトリムしない
    - level_labels: x軸ラベル（例：levels_sorted）。無ければ 0..K-1 を使う
    """
    A = np.asarray(A).astype(int).ravel()
    Y = np.asarray(Y).astype(int).ravel()
    seg = np.asarray(seg).astype(int).ravel()

    e  = np.clip(np.asarray(nuis["e_hat"]), 1e-4, 1-1e-4)
    p1 = np.asarray(nuis["p_hat1"])   # (n,K)
    p0 = np.asarray(nuis["p_hat0"])   # (n,K)
    
    K  = p1.shape[1]

    # 予測CDF（単調）
    F1 = np.cumsum(p1, axis=1)  # (n,K)
    F0 = np.cumsum(p0, axis=1)

    # IF-guard 重み
    if cap is None:
        w1 = A / e
        w0 = (1 - A) / (1 - e)
    else:
        w1 = np.minimum(1.0 / e, cap) * A
        w0 = np.minimum(1.0 / (1.0 - e), cap) * (1 - A)

    # x軸ラベル（元のカテゴリ値）を付けたい場合
    if level_labels is None:
        th = np.arange(K)
    else:
        th = np.asarray(level_labels)

    rows = []
    for g in np.unique(seg):
        idx_g = (seg == g)
        n_g = int(idx_g.sum())
        if n_g == 0:
            continue

        # overlapチェック（任意：危険なクラスはスキップしたい場合）
        n1 = int(((A == 1) & idx_g).sum())
        n0 = int(((A == 0) & idx_g).sum())
        # 必要なら下の条件でcontinue
        # if min(n1, n0) < 10:
        #     continue

        for c in range(K):
            Zc = (Y <= c).astype(float)      # 観測のI(Y<=c)
            mu1_c = F1[:, c]                 # 予測 F1(c|X)
            mu0_c = F0[:, c]                 # 予測 F0(c|X)

            # DR補正（個体別）
            psi1 = mu1_c + w1 * (Zc - mu1_c)
            psi0 = mu0_c + w0 * (Zc - mu0_c)
            F1g = psi1[idx_g].mean()
            F0g = psi0[idx_g].mean()
            F1g = float(np.clip(F1g, 0.0, 1.0))
            F0g = float(np.clip(F0g, 0.0, 1.0))
            score_c = psi1 - psi0

            # クラス内平均＆SE（EIFの分散/ n）
            tau = F1g - F0g
            phi = score_c[idx_g] - float(score_c[idx_g].mean())
            se  = float(np.sqrt(np.mean(phi**2) / n_g))
            
            rows.append({
                "seg": int(g),
                "c": int(c),
                # "threshold": th[c],
                # "n": n_g,
                # "n_treat1": n1,
                # "n_treat0": n0,
                "F1_dr": float(F1g),
                "F0_dr": float(F0g),
                "tau_c": tau,
                "se_c": se,
                "ci_low": tau - 1.96*se,
                "ci_high": tau + 1.96*se,
            })
    return pd.DataFrame(rows)

def drcdf_plot(A0, Y0, nuis, seg0, levels):
    # Y は 0..K-1 にエンコード済みのYを使う
    # levelsを用いて、欠損値（NaN）を除いた一意（ユニーク）な値のリストを作成し、それを昇順に並べ替える（元の1..5など）

    # if_c = 25
    cdf_seg = oc_dr_cdf_by_seg(
        A=A0, Y=Y0, nuis=nuis, seg=seg0,
        cap=None,                 
        level_labels=levels
    )

    print(cdf_seg) #必要なデータのみ抽出

    for g in sorted(cdf_seg["seg"].unique()):
        d = cdf_seg[cdf_seg["seg"] == g].sort_values("c")

        plt.figure()
        plt.plot(d["threshold"], d["F1_dr"], marker="o", label="満足群")
        plt.plot(d["threshold"], d["F0_dr"], marker="o", label="不満足群")
        plt.xlabel("他者推薦の点数のしきい値")
        plt.ylabel("累積確率")
        # plt.title(f"DR-CDF (seg={g})")
        plt.legend()
        plt.show()
        
    for g in sorted(cdf_seg["seg"].unique()):
        d = cdf_seg[cdf_seg["seg"] == g].sort_values("c")

        plt.figure()
        plt.plot(d["threshold"], d["tau_c"], marker="o", label="DR-CDF")
        plt.fill_between(d["threshold"], d["ci_low"], d["ci_high"], alpha=0.3, label="95%CI")
        plt.xlabel("他者推薦の点数のしきい値")
        plt.ylabel("満足群 - 不満足群")
        # plt.title(f"DR-CDF (seg={g})")
        plt.legend()
        plt.show()
        
    return cdf_seg