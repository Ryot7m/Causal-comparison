import numpy as np
from sklearn.calibration import CalibratedClassifierCV

def make_calibrated_or_base(base_estimator, method, y, default_cv=3):
    """
    モデルに対してキャリブレーションを行い、
    クラスの頻度に合わせたクロスバリデーションで調整する
    segmentation.py,aipw.pyで使用
    """
    if method not in ("isotonic", "sigmoid"):
        return base_estimator #これらを除き、キャリブレーションを行わない
    if y is None or len(y) == 0:
        return base_estimator #データ無しも同様
    _, counts = np.unique(y, return_counts=True) #目的変数の値の計算
    min_count = counts.min()
    cv = min(default_cv, int(min_count))
    if cv >= 2: #サンプル数に応じた交差検証
        return CalibratedClassifierCV(base_estimator, method=method, cv=cv)
    if len(counts) == 0:
        return base_estimator
    if cv >= 2: #サンプル数に応じた交差検証
        return CalibratedClassifierCV(base_estimator, method=method, cv=cv)
    else:
        return base_estimator