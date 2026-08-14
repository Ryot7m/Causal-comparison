import pandas as pd
from io import StringIO
from app.dantic import AnalysisRequest
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
    treatment: object
    outcome_col: str
    segment_col: str
    confounder_cols: list[str] | None = None
    
    treatment_col: str = "__treatment__"

    #欠損処理の選択
    missing_type: Literal["drop", "zero", "fill"] = "drop"
    outcome_levels: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    score_values: list[float] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    segment_missing_values: tuple = ()
    weight_cap: float = 100.0
    
    categorical_cols: list[str] = field(default_factory=list)
    fill_values: dict = field(default_factory=dict)
    
def validate_data(data: pd.DataFrame, config: AnalysisConfig):

    required = {
        config.outcome_col,
        config.segment_col,
        *config.confounder_cols,
    }
    
    if config.treatment.mode == "binary_column":
        required.add(config.treatment.column)

    elif config.treatment.mode == "quantile":
        required.add(config.treatment.source_column)

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"必要な列が存在しません: {sorted(missing)}"
        )
    

def to_analysis_config(
    request: AnalysisRequest,
) -> AnalysisConfig:
    config = AnalysisConfig(
        treatment=request.treatment,
        outcome_col=request.outcome.column,
        outcome_levels=request.outcome.levels,
        score_values=request.outcome.scores,
        segment_col=request.segment.column,
        segment_missing_values=tuple(
            request.segment.missing_values
        ),
        confounder_cols=request.covariates.columns,
        categorical_cols=(
            request.covariates.categorical_columns
        ),
        missing_type=request.missing.strategy,
        fill_values=request.missing.fill_values,
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

        "ate": result["ate"]["res"].to_dict(orient = "records"),

        "drcdf": result["drcdf"].to_dict(orient = "records"),

        "hei": {
            "score" : result["hei"]
        }
    }
    
async def estimate_service(
    file: UploadFile,
    request_config: AnalysisRequest,
):
    data = await load_csv(file)

    analysis_config = to_analysis_config(
        request_config
    )

    validate_data(
        data,
        analysis_config,
    )

    result = estimate(
        data,
        analysis_config,
    )

    return create_response(result)