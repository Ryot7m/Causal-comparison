"""
このコードは研究の流れを基に各関数を呼び出して、
それぞれの機能を繋げる役割を持つコードである
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
# import japanize_matplotlib
from workspace.segmentation import segmentation_rtn
from workspace.aipw import aipw_ate
from workspace.ateplot import ate_plot
from workspace.drcdf import drcdf_plot
from workspace.hei import hei_result 
from fastapi import FastAPI
from app.api import router

from app.config import (
    API_TITLE,
    API_VERSION
)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}

data = pd.read_csv("sample.csv", encoding="shift-jis")

data["Feature_1"] = 6 - data["Feature_1"]
data = data.fillna(0)
data_filt = data.filter(regex="Q16", axis = 1)
data_filt = 5 - data_filt
data = pd.concat([data, data_filt], axis = 1)

# 列名の変更(列名に応じて変更をしなくても可)
data = data.rename(columns={
    "満足度A":"Feature_1",
    "満足度B":"Feature_2",
    "満足度C":"Feature_12",
    "期待値":"except",
    "処置":"Treatment",
    "アウトカム変数":"Outcome"
})

outcome_col = "Outcome" 
state_col = "Feature_1"
score_rec = np.array([0, 1, 2, 3, 4],dtype= "float") 

th_sh = data[state_col].quantile(0.75)
data["Treatment"] = (data[state_col] >= th_sh).astype(int)
seg_col = "except"

#交絡因子
ftr_cols = [c for c in data.columns]
ftr_cols.remove("Outcome")
ftr_cols = [i for i in ftr_cols if i not in ['Feature_1', 'Feature_2', 'Feature_3',"Feature_4"]]
ftr_cols.remove("Treatment") #必要に応じて変数の削除
ftr_cols.remove(seg_col)

X = pd.get_dummies(data[ftr_cols], drop_first=False).values
A = data["treatment"].astype(int).values

levels = pd.Series(data[outcome_col]).dropna().unique().tolist()
levels_sorted = sorted(levels)  
# y_to_index = {lev:i for i, lev in enumerate(levels_sorted)}

outcome_levels = [1, 2, 3, 4, 5]  # 必ず意味上の順序で指定

y_to_index = {level: i for i, level in enumerate(outcome_levels)}
Y = data["Outcome"].map(y_to_index).astype(int).to_numpy()
# Y = pd.Series(data[outcome_col]).map(y_to_index).astype(int).values

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
    min_count = counts.min()
    cv = min(default_cv, int(min_count))
    if cv >= 2: #サンプル数に応じた交差検証
        return CalibratedClassifierCV(base_estimator, method=method, cv=default_cv)
    if len(counts) == 0:
        return base_estimator
    if cv >= 2: #サンプル数に応じた交差検証
        return CalibratedClassifierCV(base_estimator, method=method, cv=cv)
    else:
        return base_estimator
    
sgm = segmentation_rtn(data, seg_col, ftr_cols, A, X, Y)
ate = aipw_ate(sgm["X0"], sgm["A0"], sgm["Y0"], sgm["seg0"], 100)
ate_plot(sgm["A0"], sgm["Y0"], ate["score"], ate["nuis"], sgm["seg0"])
drcdf_plot(sgm["A0"], sgm["Y0"], ate["nuis"], sgm["seg0"], levels_sorted)
hei_result(ate["nuis"], sgm["A0"], sgm["Y0"], sgm["S0"] ,sgm["per_seg"])