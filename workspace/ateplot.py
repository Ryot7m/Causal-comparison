import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _subset_nuis(nuis, idx):
    """nuis(dict) のうち、先頭次元がnの配列だけ idx で切る"""
    out = {}
    n = len(idx)
    for k, v in nuis.items():
        if isinstance(v, np.ndarray) and v.shape[0] == n:
            out[k] = v[idx]
        elif isinstance(v, pd.Series) and len(v) == n:
            out[k] = v.to_numpy()[idx]
        else:
            out[k] = v
    return out

plt.rcParams['font.size'] = 24
def aipw_if_guard_plot(A, Y, s_values, nuis, cap, B=1000, seed=123, bins=30,
                       title=None, save_path=None):
    """
    aipw_if_guard と同じタイプの図（ブート分布 + 95%CI縦線）を1回描く。
    - A, Y は同じ長さ (n,)
    - s_values は長さK（レベル→スコア）
    - nuis は少なくとも e_hat と（mu1_hat/mu0_hat もしくは p_hat1/p_hat0）を含む想定
    """
    A = np.asarray(A).astype(int).ravel()
    Y = np.asarray(Y).astype(int).ravel()
    s_values = np.asarray(s_values, dtype=float).ravel()

    # Z_i = score(Y_i)
    Z = s_values[Y]

    # PS
    e = np.asarray(nuis["e_hat"], dtype=float).ravel()
    e = np.clip(e, 1e-4, 1 - 1e-4)

    # mu1, mu0（無ければ p_hat から作る）
    if "mu1_hat" in nuis and "mu0_hat" in nuis:
        mu1 = np.asarray(nuis["mu1_hat"], dtype=float).ravel()
        mu0 = np.asarray(nuis["mu0_hat"], dtype=float).ravel()
    else:
        # p_hat1/p_hat0 がある場合（n,K）
        p1 = np.asarray(nuis["p_hat1"], dtype=float)
        p0 = np.asarray(nuis["p_hat0"], dtype=float)
        mu1 = p1 @ s_values
        mu0 = p0 @ s_values

    # IF-guard（逆重み上限制御）
    if cap is None:
        w1 = A / e
        w0 = (1 - A) / (1 - e)
    else:
        w1 = A * np.minimum(1.0 / e, cap)
        w0 = (1 - A) * np.minimum(1.0 / (1.0 - e), cap)

    # AIPWスコア（psi）
    psi = (mu1 - mu0) + w1 * (Z - mu1) - w0 * (Z - mu0)

    # 点推定
    tau = float(np.mean(psi))

    # ブートストラップ（psiを再標本化）
    rng = np.random.default_rng(seed)
    n = len(psi)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        ii = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(psi[ii]))

    # 95% CI（パーセンタイル）
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    # 描画
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(boots, bins=bins, density=True)
    plt.axvline(ci_low, color="green")
    plt.axvline(tau, color="red")
    plt.axvline(ci_high, color="orange")
    plt.xlabel("ATE")
    plt.ylabel("density")
    plt.title(title or f"ATEと95%信頼区間(IF-guard cap={cap}, B={B})")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()

    return {
        "ate": tau,
        "ci_low_95": float(ci_low),
        "ci_high_95": float(ci_high),
        "B": int(B),
        "cap": cap,
        "n": int(n),
    }

def plot_if_guard_seg(A, Y, s_values, nuis, seg, cap, B=1000, seed=123, bins=30,
                         exclude_segs=(-1,), save_dir=None):
    """
    segごとに aipw_if_guard_plot を回して図を出す。
    """
    A = np.asarray(A).ravel()
    Y = np.asarray(Y).ravel()
    seg = np.asarray(seg).ravel()

    uniq = np.unique(seg)
    uniq = [g for g in uniq if g not in exclude_segs]

    results = []
    for j, g in enumerate(uniq):
        idx = (seg == g)
        if idx.sum() == 0:
            continue

        nuis_g = _subset_nuis(nuis, idx)

        title = f"ATEと95%信頼区間 正確性 クラス{int(g)}"
        save_path = None
        if save_dir is not None:
            save_path = f"{save_dir}/if_guard_seg{int(g)}.png"

        out = aipw_if_guard_plot(
            A[idx], Y[idx], s_values, nuis_g,
            cap=cap, B=B, seed=seed + j, bins=bins,
            title=title, save_path=save_path
        )
        out["seg"] = int(g)
        out["n_treated"] = int(np.sum(A[idx] == 1))
        out["n_control"] = int(np.sum(A[idx] == 0))
        results.append(out)

    return pd.DataFrame(results).sort_values("seg").reset_index(drop=True)


def ate_plot(A0, Y0, score, nuis, seg0, cap):
    res_if_seg = plot_if_guard_seg(
        A0, Y0, score, nuis, seg0,
        cap, B=1000, seed=123, bins=30
)