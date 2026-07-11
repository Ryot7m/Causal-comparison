import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import auc
import japanize_matplotlib
from segmentation import segmentation_rtn
from aipw import aipw_ate

data = pd.read_csv("使用データ.csv", encoding="shift-jis")

data["Q4_1"] = 6 - data["Q4_1"]
data = data.fillna(0)
data_filt = data.filter(regex="Q16", axis = 1)
data_filt = 5 - data_filt
data = pd.concat([data, data_filt], axis = 1)

# 列名の変更
data = data.rename(columns={
    "Q1": "満足度",
    "Q2_7": "正確性",
    "Q2_8": "更新頻度",
    "Q2_9": "豊富さ",
    "Q2_10": "詳細さ",
    "Q4_1": "他者推薦",
})

outcome_col = "他者推薦" 
state_col = "詳細さ"
score_rec = np.array([0, 1, 2, 3, 4],dtype= "float") 

th_sh = data[state_col].quantile(0.75)
data["treatment"] = (data[state_col] >= th_sh).astype(int)
seg_col = "Q7_4"

#交絡因子
ftr_cols = [c for c in data.columns]
ftr_cols.remove("他者推薦")
ftr_cols = [i for i in ftr_cols if i not in ['正確性', '更新頻度', '詳細さ',"豊富さ"]]
ftr_cols.remove("treatment")
ftr_cols.remove(seg_col)
ftr_cols.remove("SQ2")

X = pd.get_dummies(data[ftr_cols], drop_first=False).values
A = data["treatment"].astype(int).values

levels = pd.Series(data[outcome_col]).dropna().unique().tolist()
levels_sorted = sorted(levels)  
y_to_index = {lev:i for i, lev in enumerate(levels_sorted)}
Y = pd.Series(data[outcome_col]).map(y_to_index).astype(int).values

K = len(levels_sorted)

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
    min_count = counts.min() if len(counts) else 0
    if min_count >= 2: #サンプル数に応じた交差検証
        return CalibratedClassifierCV(base_estimator, method=method, cv=default_cv)
    else:
        return base_estimator
    
segmentation_rtn(data, seg_col, ftr_cols, A, X, Y)
aipw_ate(X1, A0, Y0, seg0)