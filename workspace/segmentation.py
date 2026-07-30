import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from workspace.calibrated import make_calibrated_or_base

def make_seg_from_cuts(S, cut1, cut2):
    """seg0: S<=cut1, seg1: cut1<S<=cut2, seg2: S>cut2"""
    S = np.asarray(S)
    return (S > cut1).astype(int) + (S > cut2).astype(int)

def fit_ps_oof(X, A, n_splits=5, random_state=42, C=0.5,
    class_weight= None,
    calibration="isotonic",
    max_iter=3000,eps=1e-4):
    """
    PSをアウトカム無しで推定（out-of-fold）。
    セグメント内でサンプルが少ない場合は自動で分割数を落とす。
    """
    A = np.asarray(A).astype(int).ravel()
    X = np.asarray(X, dtype=float)
    n = len(A)

    n1 = int(A.sum())
    n0 = int(n - n1)
    k = min(n_splits, n1, n0)
    if k < 2:
        p = float(A.mean())
        return np.clip(np.full(n, p, dtype=float), eps, 1-eps)

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    e_raw = np.empty(n, dtype=float)

    base_ps = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            C=float(C),
            class_weight=class_weight
        ))
    ])
    
    ps_hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, random_state=123)
    ps_rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, class_weight=None, random_state=123, n_jobs=-1)

    for tr, te in cv.split(X, A):
        ps_model = make_calibrated_or_base(base_ps, calibration, A[tr], default_cv=3)
        ps_model.fit(X[tr], A[tr])
        e_raw[te] = ps_model.predict_proba(X[te])[:, 1]

    # clip（極端値で重みが爆発するのを防ぐ）
    e_hat = np.clip(e_raw, 1e-2, 1 - 1e-2)
    return e_hat

def ess(w):
    # w = np.asarray(w, dtype=float)
    return (w.sum() ** 2) / (np.square(w).sum() + 1e-12)

def weighted_smd(X, A, e_hat):
    X = np.asarray(X, dtype=float)
    A = np.asarray(A).astype(int).ravel()
    e = np.asarray(e_hat, dtype=float)

    Xt = X[A == 1]
    Xc = X[A == 0]
    wt = 1.0 / e[A == 1]
    wc = 1.0 / (1.0 - e[A == 0])

    mt = np.average(Xt, axis=0, weights=wt)
    mc = np.average(Xc, axis=0, weights=wc)

    vt = np.average((Xt - mt) ** 2, axis=0, weights=wt)
    vc = np.average((Xc - mc) ** 2, axis=0, weights=wc)

    smd = (mt - mc) / np.sqrt((vt + vc) / 2.0 + 1e-12)
    abs_smd = np.abs(smd)

    return float(np.nanmax(abs_smd)), float(np.nanmean(abs_smd))

def rank_cuts_design_optimal(S, A, X, min_total_per_seg, min_each_treat_per_seg,
                             n_splits_ps=5, random_state=42, cap_weight=None,ps_kwargs=None,):
    """
    アウトカムを使わず、design（overlap/balance/ESS）でcutをランキング
    """
    S = np.asarray(S)
    A = np.asarray(A).astype(int).ravel()
    X = np.asarray(X, dtype=float)

    # 欠損（NaN）はここでは除外してcut探索（必要なら後で別扱い）
    ok = ~pd.isna(S)
    S0, A0, X0 = S[ok], A[ok], X[ok]

    uniq = np.sort(np.unique(S0))
    cuts = uniq[:-1]  # 最大値は境界にしない
    
    ps_kwargs = {} if ps_kwargs is None else dict(ps_kwargs)

    rows = []
    for i, c1 in enumerate(cuts):
        for c2 in cuts[i+1:]:
            seg = make_seg_from_cuts(S0, c1, c2)

            # セグメントの成立条件（両群がいて十分な数）
            seg_ok = True
            seg_info = []
            worst_max_smd = 0.0
            worst_extreme = 0.0
            worst_max_w = 0.0
            min_ess_ratio = 1.0

            for g in [0, 1, 2]:
                idx = (seg == g)
                n = int(idx.sum())
                n1 = int(((A0 == 1) & idx).sum())
                n0c = int(((A0 == 0) & idx).sum())

                if (n < min_total_per_seg) or (n1 < min_each_treat_per_seg) or (n0c < min_each_treat_per_seg):
                    seg_ok = False
                    break

                e_hat = fit_ps_oof(X0[idx], A0[idx], n_splits=n_splits_ps, random_state=random_state, **ps_kwargs)
                if e_hat is None:
                    seg_ok = False
                    break

                # overlap/weights
                extreme = float(((e_hat < 0.05) | (e_hat > 0.95)).mean())
                w = np.where(A0[idx] == 1, 1.0 / e_hat, 1.0 / (1.0 - e_hat))
                if cap_weight is not None:
                    w = np.minimum(w, cap_weight)
                ess_ratio = float(ess(w) / len(w))
                max_w = float(w.max())

                # balance
                max_smd, mean_smd = weighted_smd(X0[idx], A0[idx], e_hat)

                worst_max_smd = max(worst_max_smd, max_smd)
                worst_extreme = max(worst_extreme, extreme)
                worst_max_w = max(worst_max_w, max_w)
                min_ess_ratio = min(min_ess_ratio, ess_ratio)

                seg_info.append((g, n, n1, n0c, max_smd, extreme, ess_ratio, max_w))

            if not seg_ok:
                continue

            rows.append({
                "cut1": c1,
                "cut2": c2,
                "worst_max_abs_SMD": worst_max_smd,
                "worst_extreme_ps_rate": worst_extreme,
                "min_ess_ratio": min_ess_ratio,
                "worst_max_weight": worst_max_w,
                "seg_info": seg_info
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # “minimax設計最適”の並び
    out = out.sort_values(
        ["worst_max_abs_SMD", "worst_extreme_ps_rate", "min_ess_ratio", "worst_max_weight","cut1"],
        ascending=[True, True, False, True, True]
    ).reset_index(drop=True)
    return out

def weighted_smd_detail(X, A, e_hat, cap_weight=100, feature_names=None):
    
    if hasattr(X, "to_numpy"):
        if feature_names is None:
            feature_names = list(X.columns)
        Xv = X.to_numpy()
    else:
        Xv = np.asarray(X)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(Xv.shape[1])]

    A = np.asarray(A).astype(int).ravel()
    e = np.clip(np.asarray(e_hat).ravel(), 1e-4, 1-1e-4)

    if cap_weight is None:
        wt = A / e
        wc = (1 - A) / (1 - e)
    else:
        wt = A * np.minimum(1.0 / e, cap_weight)
        wc = (1 - A) * np.minimum(1.0 / (1 - e), cap_weight)

    def wmean(x, w):
        m = np.isfinite(x) & np.isfinite(w)
        return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.sum() else np.nan

    def wvar(x, w):
        m = np.isfinite(x) & np.isfinite(w)
        if m.sum() == 0:
            return np.nan
        mu = np.sum(x[m] * w[m]) / np.sum(w[m])
        return float(np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m]))

    rows = []
    p = Xv.shape[1]
    xt = Xv[A == 1, :]
    xc = Xv[A == 0, :]
    wt1 = wt[A == 1]
    wc0 = wc[A == 0]

    for j in range(p):
        mt = wmean(xt[:, j], wt1)
        mc = wmean(xc[:, j], wc0)
        vt = wvar(xt[:, j], wt1)
        vc = wvar(xc[:, j], wc0)
        denom = np.sqrt(0.5 * (vt + vc) + 1e-12)
        smd = (mt - mc) / denom if np.isfinite(mt) and np.isfinite(mc) and denom > 0 else np.nan

        rows.append({
            "feature": str(feature_names[j]),
            "smd": smd,
            "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
            "mean_treated_w": mt,
            "mean_control_w": mc,
            "var_treated_w": vt,
            "var_control_w": vc,
        })

    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False, na_position="last")

def ess_class(w: np.ndarray):
    w = np.asarray(w, dtype=float)
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    return float((s1 * s1) / s2) if s2 > 0 else np.nan

def segmentation_rtn(S, feature_names, A, X, treat):
      # 0埋めしている場合は欠損扱いへ

    ps_kwargs_design = dict(
        C=0.5,               # 小さめほど正則化が強い（極端PSを抑えやすい）
        calibration= "isotonic",
    )
    cap_weight_design = 100

    rank_design = rank_cuts_design_optimal(
        S=S, A=A, X=X,
        min_total_per_seg=200, min_each_treat_per_seg=10,
        cap_weight=cap_weight_design,
        ps_kwargs=ps_kwargs_design
    )

    print(rank_design.head(10))

    best = rank_design.iloc[0]
    cut1, cut2 = best["cut1"], best["cut2"]
    seg_opt = make_seg_from_cuts(S, cut1, cut2)
    print("best cuts:", cut1, cut2)
    print("seg info:", best["seg_info"])
    print(pd.crosstab(seg_opt, treat))

    # --- rank_cuts_design_optimal と同じ “欠損除外” で確認 ---

    # X は numpy なので列名が消えている。dummiesの列名を復元する

    seg0 = make_seg_from_cuts(S, cut1, cut2)
    per_seg_summary = []
    all_rows = []

    for g in [0, 1, 2]:
        idx_g = (seg0 == g)
        n_g = int(idx_g.sum())
        n1 = int(((A == 1) & idx_g).sum())
        n0 = int(((A == 0) & idx_g).sum())
        print(f"\n--- seg={g} | n={n_g} | treated={n1} | control={n0} ---")

        if n_g == 0 or min(n1, n0) < 2:
            print("skip (too small)")
            continue
        
        # 3) seg0に切る
        e_hat = fit_ps_oof(X[idx_g], A[idx_g], C=0.5, calibration="isotonic", eps=1e-2)
        # e_hat_g_g = e_hat[idx_g]
        # 4) SMD詳細（上位表示）
        df_smd = weighted_smd_detail(X[idx_g], A[idx_g], e_hat, cap_weight=None, feature_names = feature_names)
        max_abs = float(df_smd["abs_smd"].iloc[0])
        mean_abs = float(df_smd["abs_smd"].mean())
        p95_abs = float(df_smd["abs_smd"].quantile(0.95))
        top_feat = str(df_smd["feature"].iloc[0])

        extreme_ps_rate = float(((e_hat < 0.05) | (e_hat > 0.95)).mean())

        a_g = A[idx_g]
        wt = np.minimum(1.0 / e_hat[a_g == 1], 100)
        wc = np.minimum(1.0 / (1.0 - e_hat[a_g == 0]), 100)

        ess_t = ess_class(wt)
        ess_c = ess_class(wc)
        ess_ratio_t = float(ess_t / len(wt)) if len(wt) > 0 else np.nan
        ess_ratio_c = float(ess_c / len(wc)) if len(wc) > 0 else np.nan
        ess_ratio_min = float(np.nanmin([ess_ratio_t, ess_ratio_c]))

        per_seg_summary.append({
            "seg": int(g),
            "n": n_g,
            "n_treated": n1,
            "n_control": n0,
            "seg_max_abs_SMD": max_abs,
            "seg_mean_abs_SMD": mean_abs,
            "seg_p95_abs_SMD": p95_abs,
            "seg_top_feature": top_feat,
            "extreme_ps_rate": extreme_ps_rate,
            "ess_ratio_min": ess_ratio_min,
        })
        
        
        # overall集約用（feature×seg を保持）
        tmp = df_smd[["feature", "smd", "abs_smd"]].copy()
        tmp["seg"] = int(g)
        tmp["n_seg"] = n_g
        all_rows.append(tmp)

    per_seg_df = pd.DataFrame(per_seg_summary).sort_values("seg")
    print(per_seg_df)

    # all_smd = pd.concat(all_rows, ignore_index=True)

    # # ===== overall の作り方（おすすめ3指標） =====
    # # 1) overall_max_abs_SMD : どこかのseg×featureで最悪のSMD（=最も保守的）
    # overall_max_abs_SMD = float(all_smd["abs_smd"].max())

    # # 2) overall_mean_abs_SMD (segサイズで重み付けした平均) : “全体としてどれくらいズレているか”
    # overall_mean_abs_SMD = float(np.average(all_smd["abs_smd"], weights=all_smd["n_seg"]))

    # # 3) overall_p95_abs_SMD : 外れ値1本に引っ張られにくい指標
    # overall_p95_abs_SMD = float(all_smd["abs_smd"].quantile(0.95))
    
    return  {
        # "X0" : X,
        # "A0" : A,
        # "Y0" : Y,
        # "S0" : S,
        "seg0" : seg0,
        "per_seg" : per_seg_summary,
        "cut1" : cut1,
        "cut2" : cut2
    }