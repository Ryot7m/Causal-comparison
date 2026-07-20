import pandas as pd
from io import StringIO

from workspace.aipw import aipw_ate
from workspace.segmentation import segmentation_rtn
from workspace.ateplot import ate_plot
from workspace.drcdf import drcdf_plot
from workspace.hei import hei_result
from fastapi import UploadFile
from app.config import (
    OUTCOME_COL,
    TREATMENT_COL,
    EXPERT_COL,
    EXCLUDE_COLUMNS,
)

async def load_csv(file: UploadFile) -> pd.DataFrame:

    contents = await file.read()

    return pd.read_csv(
        StringIO(contents.decode("utf-8"))
    )
    
def validate_data(
    data: pd.DataFrame
):

    if OUTCOME_COL not in data.columns:
        raise ValueError(
            f"{OUTCOME_COL} が存在しません。"
        )

    if TREATMENT_COL not in data.columns:
        raise ValueError(
            f"{TREATMENT_COL} が存在しません。"
        )
    
def preprocess(
    data
):

    ftr_cols = [
        c for c in data.columns
        if c not in EXCLUDE_COLUMNS
    ]

    X = pd.get_dummies(data[ftr_cols], drop_first=False).values
    A = data[TREATMENT_COL].astype(int).values

    levels = pd.Series(data[OUTCOME_COL]).dropna().unique().tolist()
    levels_sorted = sorted(levels)  
    y_to_index = {lev:i for i, lev in enumerate(levels_sorted)}
    Y = pd.Series(data[OUTCOME_COL]).map(y_to_index).astype(int).values
    
    return {
        "feature" : ftr_cols,
        "A" : A,
        "X" : X,
        "Y" : Y,
        "level" : levels_sorted
    }
    
def estimate(data, prep):

    sgm = segmentation_rtn(data, EXPERT_COL, prep["feature"], prep["A"], prep["X"], prep["Y"])
    ate = aipw_ate(sgm["X1"], sgm["A0"], sgm["Y0"], sgm["seg0"])
    ate_plot(sgm["A0"], sgm["Y0"], ate["nuis"], ate["score"], sgm["seg0"])
    dr_cdf = drcdf_plot(sgm["A0"], sgm["Y0"], ate["nuis"], sgm["seg0"], prep["level"])
    hei = hei_result(ate["nuis"], sgm["A0"], sgm["Y0"], sgm["S0"] ,sgm["per_seg"])
    
    return {

        "segment": sgm,
        "ate": ate,
        "drcdf": dr_cdf,
        "hei" : hei
    }

def create_response(result):
    return {

        "segment": {

            "cut1": result["segment"]["cut1"],
            "cut2": result["segment"]["cut2"]

        },

        "ate": result["ate"]["res"],

        "drcdf": result["drcdf"],

        "hei": result["hei"]
    }
    
async def estimate_service(file: UploadFile):

    data = await load_csv(file)

    validate_data(data)

    prep = preprocess(data)
    
    result = estimate(data, prep)

    return create_response(result)