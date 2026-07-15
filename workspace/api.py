from fastapi import FastAPI
import pandas as pd

from workspace.aipw import aipw_ate
from workspace.segmentation import segmentation_rtn

app = FastAPI()

@app.post("/estimate")
def estimate():

    data = pd.read_csv("sample.csv")

    sgm = segmentation_rtn(data, seg_col, ftr_cols, A, X, Y)
    ate = aipw_ate(sgm["X1"], sgm["A0"], sgm["Y0"], sgm["seg0"])

    return {
        "ATE": ate
    }