import pandas as pd
from io import StringIO
from dataclasses import dataclass, field
from workspace.aipw import aipw_ate
from workspace.segmentation import segmentation_rtn
from workspace.ateplot import ate_plot
from workspace.drcdf import drcdf_plot
from workspace.hei import hei_result
from workspace.preprocess import pre_analysis
from typing import Literal
from fastapi import UploadFile

async def load_csv(file: UploadFile) -> pd.DataFrame:

    contents = await file.read()

    return pd.read_csv(
        StringIO(contents.decode("utf-8"))
    )
@dataclass
class AnalysisConfig:
    treatment_col: str
    outcome_col: str
    segment_col: str
    confounder_cols: list[str] | None = None
    state_col: str | None = None
    threshold: float | None = None
    exclude_cols: list[str] = field(default_factory=list)
    exclude_conditions: list[str] = field(default_factory=list)
    treatment_source_col: str | None = None

    #欠損処理の選択
    missing_type: Literal["drop", "zero"] = "zero"
    zero_fill: list[str] | None = None

    outcome_levels: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    score_values: list[float] = field(
        default_factory=lambda: [1, 2, 3, 4, 5]
    )
    reverse_score_max: dict[str, float] = field(default_factory=dict)
    segment_missing_values: tuple = ()
    weight_cap: float = 100.0
    
def validate_data(data: pd.DataFrame, config: AnalysisConfig):

    required = {
        config.outcome_col,
        config.segment_col,
        config.state_col,
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"必要な列が存在しません: {sorted(missing)}"
        )
    

def config_name():
    config = AnalysisConfig(
        treatment_col="Treatment",
        outcome_col="Q4_1",
        segment_col="Q7_4",
        state_col="Q2_9",
        threshold=0.75,
        confounder_cols=None,
        reverse_score_max={"Q4_1": 6},
        exclude_conditions=["Q2_*"],
        exclude_cols=[]
    )
    
    return config
    
def estimate(data, config):

    prcs = pre_analysis(data, config) 
    sgm = segmentation_rtn(prcs["S"], prcs["ftr"], prcs["A"], prcs["X"], config)
    ate = aipw_ate(prcs["X"], prcs["A"], prcs["Y"], sgm["seg0"], prcs["score"], config.weight_cap)
    ate_plot(prcs["A"], prcs["Y"], ate["score"], ate["nuis"], sgm["seg0"], config.weight_cap)
    dr_cdf = drcdf_plot(prcs["A"], prcs["Y"], ate["nuis"], sgm["seg0"], prcs["level"])
    hei = hei_result(ate["nuis"], prcs["A"], prcs["Y"], prcs["S"] ,sgm["per_seg"], prcs["score"])
    
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

    prep = config_name()
    
    result = estimate(data, prep)

    return create_response(result)