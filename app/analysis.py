import pandas as pd
import numpy as np
from io import StringIO
from app.dantic import AnalysisRequest
from dataclasses import dataclass, field
from workspace.aipw import aipw_ate
from workspace.segmentation import segmentation_rtn
from workspace.drcdf import oc_dr_cdf_by_seg
from workspace.hei import hei_result
from workspace.preprocess import pre_analysis
from typing import Literal
from fastapi import UploadFile
from sklearn.model_selection import train_test_split
from workspace.segmentation import make_seg_from_cuts, select_segment_cuts



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
    idx_design, idx_est = train_test_split(
        np.arange(len(prcs["A"])),
        test_size=0.5,
        random_state=123,
        stratify=prcs["A"]
    ) 
    
    idx_s = prcs["S"][idx_est]
    idx_a = prcs["A"][idx_est]
    idx_x = prcs["X"][idx_est]
    idx_y = prcs["Y"][idx_est]
    
    slc = select_segment_cuts(prcs["S"][idx_design], prcs["A"][idx_design], prcs["X"][idx_design], config)
    # sgm = segmentation_rtn(prcs["S"][idx_design], prcs["ftr"], prcs["A"][idx_design], prcs["X"][idx_design], config)
    seg_est = make_seg_from_cuts(prcs["S"][idx_est], slc["cut1"], slc["cut2"])
    ate = aipw_ate(idx_x, idx_a, idx_y, seg_est, prcs["score"], config.weight_cap)
    dr_cdf = oc_dr_cdf_by_seg(idx_a, idx_y, ate["nuis"], seg_est, None, prcs["level"])
    # hei = hei_result(ate["nuis"], idx_a, idx_y, idx_s ,sgm["per_seg"], prcs["score"])
    hei = None
    
    return {

        "segment": slc,
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

        # create_response()
        "hei": None if result["hei"] is None else {"score": result["hei"]}
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