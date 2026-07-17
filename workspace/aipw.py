import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from main import make_calibrated_or_base

def ensure_numpy(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    return np.asarray(x) #numpy配列に変換

def clip_eps(arr, eps=1e-4):
    return np.clip(arr, eps, 1 - eps) #指定範囲でクリッピング

def proba_aligned(estimator, X_va, K):
    """
    確率予測(predict_proba) が (n,k_obs) でも K 列に整列。
    欠落クラスは0で埋め、行正規化（全ゼロ行は一様）。
    """
    p_raw = estimator.predict_proba(X_va) #順序回帰モデルによる予測確率を出力し、各クラスの確率値の行列を取得
    if hasattr(estimator, "classes_"):
        cls = np.asarray(estimator.classes_, dtype=int)
    elif hasattr(estimator, "calibrated_classifiers_"):
        cls = np.asarray(estimator.calibrated_classifiers_[0].classes_, dtype=int)
    else:
        raise RuntimeError("Estimator has no classes_.")
    out = np.zeros((X_va.shape[0], K), dtype=float) #サンプル数×クラス数の行列
    out[:, cls] = p_raw #クラスの配置
    s = out.sum(axis=1, keepdims=True)              
    np.divide(out, s, out=out, where=(s > 0)) #行ごとの確率を正規化      
    zero_rows = (s[:, 0] == 0)                      
    if np.any(zero_rows):
        out[zero_rows, :] = 1.0 / K #ゼロ確率のサンプルに対する一様分布
    return out

def monotone_cdf_rows(p_hat):
    """
    (n,K) の順序確率 → 累積分布関数 → 行ごとに Isotonic(PAV) で単調化 → 差分で確率に復元 → 正規化。
    """
    n, K = p_hat.shape #カテゴリ確率
    F  = np.cumsum(np.clip(p_hat, 1e-6, 1.0), axis=1)  # 行ごとの累積和
    Fm = np.zeros_like(F)
    x  = np.arange(K)
    for i in range(n):
        ir = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0) #0以上1以内で単調回帰を行う
        Fm[i, :] = ir.fit_transform(x, F[i, :]) #累積分布を単調増加
    p_star = np.diff(np.hstack([np.zeros((n,1)), Fm]), axis=1)  # 差分で確率に
    p_star = np.clip(p_star, 1e-6, 1.0)
    p_star = p_star / p_star.sum(axis=1, keepdims=True) #正規化
    return p_star #これらの処理を行い、順序一貫性のあるカテゴリ変数の出力

def fit_cf_oc(X, A, Y, s_values, n_splits: int = 5, max_iter: int = 500, random_state: int = 0,):
    """
    クロスフィット＋（足りなければcv調整）確率較正 ＋
    K列整列（クラス欠落対策）＋「累積→単調化→差分復元」で順序一貫の確率を返す。
    """
    X = ensure_numpy(X)
    A = ensure_numpy(A).astype(int).ravel()
    Y = ensure_numpy(Y).astype(int).ravel()

    n, p = X.shape
    uniq = np.unique(Y)
    K = len(uniq)
    assert (uniq == np.arange(K)).all() #連続した整数の確認
    s_values = np.asarray(s_values, dtype=float)
    assert len(s_values) == K and np.all(np.diff(s_values) > 0) #順序カテゴリの値とスコアが一致しているか

    e_hat  = np.zeros(n, dtype=float)
    p_hat1 = np.zeros((n, K), dtype=float)
    p_hat0 = np.zeros((n, K), dtype=float)
    mu1_hat = np.zeros(n, dtype=float)
    mu0_hat = np.zeros(n, dtype=float)
    folds = np.zeros(n, dtype=int)
    
    # 傾向スコアモデルの設定
    base_ps = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=1500, solver="lbfgs",C=0.5, class_weight= "balanced"))
    ]) 
    
    ps_hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, random_state=123)
    ps_rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, class_weight="balanced", random_state=123, n_jobs=-1)

    ps_calibration = "sigmoid"

    # 多クラスロジット＋順序一貫な確率分布による形状制約モデルの設定
    base_mn = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=1500, multi_class="multinomial", solver="lbfgs"))
    ])
    outcome_calibration = "isotonic"
    
    # クロスバリデーション
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_id, (tr, va) in enumerate(kf.split(X)):
        folds[va] = fold_id
        X_tr, X_va = X[tr], X[va]
        A_tr, A_va = A[tr], A[va]
        Y_tr, Y_va = Y[tr], Y[va]

        # 傾向スコアの予測
        ps_model = make_calibrated_or_base(base_ps, ps_calibration, A_tr, default_cv=3)
        ps_model.fit(X_tr, A_tr)
        e_hat[va] = ps_model.predict_proba(X_va)[:, 1]
        
        #多クラスロジット＋順序一貫な確率分布による形状制約による予測
        X_tr_p = np.c_[X_tr, A_tr]; X_va_p1 = np.c_[X_va, np.ones(len(X_va))]
        X_va_p0 = np.c_[X_va, np.zeros(len(X_va))]
        mn = make_calibrated_or_base(base_mn, outcome_calibration, Y_tr, default_cv=3)
        mn.fit(X_tr_p, Y_tr)
        p1 = proba_aligned(mn, X_va_p1, K)
        p0 = proba_aligned(mn, X_va_p0, K)
        
        # 確率の整列と単調化
        p_hat1[va, :] = p1
        p_hat0[va, :] = p0
        
    # 順序一貫な確率の取得（累積→単調化→差分復元）
    p_hat1 = monotone_cdf_rows(p_hat1)
    p_hat0 = monotone_cdf_rows(p_hat0)
    mu1_hat = (p_hat1 @ s_values.reshape(-1, 1)).ravel()
    mu0_hat = (p_hat0 @ s_values.reshape(-1, 1)).ravel()

    e_hat = clip_eps(e_hat, 1e-4)

    return dict(
        e_hat=e_hat, p_hat1=p_hat1, p_hat0=p_hat0,
        mu1_hat=mu1_hat, mu0_hat=mu0_hat, folds=folds,
        s_values=s_values, K=K
    )
    
def oc_aipw_ate(A, Y, s_values, nuisance, seg, cap = 100):
    seg = ensure_numpy(seg).astype(int)
    A = ensure_numpy(A).astype(int).ravel()
    Y = ensure_numpy(Y).astype(int).ravel()
    s_values = np.asarray(s_values, dtype=float)
    Z = s_values[Y]
    e = nuisance["e_hat"] # 傾向スコア
    man = nuisance["mu1_hat"] # 満足群
    hum = nuisance["mu0_hat"] # 不満足群
    
    if cap is None:
        w1 = A / e
        w0 = (1 - A) / (1 - e)
    else:
        w1 = A * np.minimum(1.0 / e, cap)
        w0 = (1 - A) * np.minimum(1.0 / (1.0 - e), cap)
    
    score = (man - hum) + w1*(Z - man) - w0*(Z - hum) # AIPW（拡張逆確率重み推定量）の計算
    lst = []
    for i in np.unique(seg):
        id = seg == i
        tau = float(score[id].mean()) #ATE（平均処置効果）の計算
        # 標準誤差の計算
        phi = score[id] - tau # 影響関数
        se  = float(np.sqrt(np.mean(phi**2)/len(phi)))
        lst.append([i, len(phi), tau, se, tau - 1.96*se, tau + 1.96*se])
    return pd.DataFrame(lst, columns = ["cls", "clsnum", "ate", "se", "95ci_low", "95ci_high"])

def aipw_ate(X1, A0, Y0, seg0):
    score = np.asarray([0, 1, 2, 3, 4])

    nuis = fit_cf_oc(X1, A0, Y0, score, n_splits=5, random_state=123)
    res_ate = oc_aipw_ate(A0, Y0, score, nuis, seg0, cap = None)
    print(f"AIPW-ATE{res_ate}")
    
    return {
        "score" : score,
        "nuis" : nuis,
        "res" : res_ate
    }