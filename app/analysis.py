import pandas as pd
from io import StringIO

from workspace.aipw import aipw_ate
from workspace.segmentation import segmentation_rtn
from workspace.ateplot import ate_plot
from workspace.drcdf import drcdf_plot
from workspace.hei import hei_result

async def load_csv(file: UploadFile) -> pd.DataFrame:

    contents = await file.read()

    return pd.read_csv(
        StringIO(contents.decode("utf-8"))
    )
    
def validate_data(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str
):

    if outcome_col not in df.columns:
        raise ValueError(
            f"{outcome_col} が存在しません。"
        )

    if treatment_col not in df.columns:
        raise ValueError(
            f"{treatment_col} が存在しません。"
        )
    
def preprocess(
    data,
    outcome_col,
    treatment_col
):

    ftr_cols = [
        c for c in data.columns
        if c not in [
            outcome_col,
            treatment_col
        ]
    ]

    seg_col = "Outcome"

    X = pd.get_dummies(data[ftr_cols], drop_first=False).values
    A = data[treatment_col].astype(int).values

    levels = pd.Series(data[outcome_col]).dropna().unique().tolist()
    levels_sorted = sorted(levels)  
    y_to_index = {lev:i for i, lev in enumerate(levels_sorted)}
    Y = pd.Series(data[outcome_col]).map(y_to_index).astype(int).values

    sgm = segmentation_rtn(data, seg_col, ftr_cols, A, X, Y)
    ate = aipw_ate(sgm["X1"], sgm["A0"], sgm["Y0"], sgm["seg0"])
    ate_plot(sgm["A0"], sgm["Y0"], ate["nuis"], ate["score"], sgm["seg0"])
    dr_cdf = drcdf_plot(sgm["A0"], sgm["Y0"], ate["nuis"], sgm["seg0"], levels_sorted)
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