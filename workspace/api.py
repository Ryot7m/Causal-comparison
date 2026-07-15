from fastapi import FastAPI
import pandas as pd

from workspace.aipw import aipw_ate
from workspace.segmentation import segmentation_rtn

app = FastAPI()

@app.post("/estimate")
def estimate():

    data = pd.read_csv("sample.csv")
    
    seg_col = "Outcome"

    outcome_col = "Outcome"
    ftr_cols = [c for c in data.columns]
    ftr_cols.remove("Outcome")
    ftr_cols = [i for i in ftr_cols if i not in ['Feature_1', 'Feature_2', 'Feature_3',"Feature_4"]]
    ftr_cols.remove("treatment") #必要に応じて変数の削除
    ftr_cols.remove(seg_col)

    X = pd.get_dummies(data[ftr_cols], drop_first=False).values
    A = data["treatment"].astype(int).values

    levels = pd.Series(data[outcome_col]).dropna().unique().tolist()
    levels_sorted = sorted(levels)  
    y_to_index = {lev:i for i, lev in enumerate(levels_sorted)}
    Y = pd.Series(data[outcome_col]).map(y_to_index).astype(int).values

    sgm = segmentation_rtn(data, seg_col, ftr_cols, A, X, Y)
    ate = aipw_ate(sgm["X1"], sgm["A0"], sgm["Y0"], sgm["seg0"])

    return {
        "ATE": ate["res"]
    }