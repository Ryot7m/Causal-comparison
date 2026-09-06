"""
このコードは研究の流れを基に各関数を呼び出して、
それぞれの機能を繋げる役割を持つコードである
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# import japanize_matplotlib
from workspace.segmentation import segmentation_rtn
from workspace.aipw import aipw_ate
from workspace.ateplot import ate_plot
from workspace.drcdf import oc_dr_cdf_by_seg
from workspace.drcdfplot import drcdf_plot
from workspace.hei import hei_result 
from workspace.preprocess import pre_analysis
from typing import Literal

data = pd.read_csv("sample.csv", encoding="shift-jis")
data = data.copy()
data["Q4_1"] = (6 - pd.to_numeric(data["Q4_1"]))

from dataclasses import dataclass, field
from typing import Literal




@dataclass(frozen=True)
class ResearchTreatment:
    mode: Literal["quantile"]
    source_column: str
    quantile: float = 0.75
    treated_when: Literal[
        "ge",
        "gt",
        "le",
        "lt"
    ] = "ge"


@dataclass
class ResearchConfig:
    treatment: ResearchTreatment
    outcome_col: str
    segment_col: str
    confounder_cols: list[str]

    treatment_col: str = "Treatment"
    missing_type: Literal[
        "drop",
        "zero",
        "fill",
    ] = "zero"

    outcome_levels: list = field(
        default_factory=lambda: [1, 2, 3, 4, 5]
    )
    score_values: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    segment_missing_values: tuple = ()
    weight_cap: float = 100.0
    categorical_cols: list[str] = field(
        default_factory=list
    )
    fill_values: dict = field(
        default_factory=dict
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


config = ResearchConfig(
    treatment=ResearchTreatment(mode="quantile", source_column="Q2_9", quantile=0.75, treated_when="ge"),
    outcome_col="Q4_1",
    segment_col="Q7_4",
    confounder_cols=["SQ1","SQ3","SQ8","Q5_2","Q12_2","Q14_2","Q28","Q29","Q30","Q31","Q38","Q39","Q41"],
    categorical_cols=[],
    missing_type="zero",
    fill_values={}
)

prcs = pre_analysis(data, config) 
sgm = segmentation_rtn(prcs["S"], prcs["ftr"], prcs["A"], prcs["X"], config)
ate = aipw_ate(prcs["X"], prcs["A"], prcs["Y"], sgm["seg0"], prcs["score"], config.weight_cap)
ate_plot(prcs["A"], prcs["Y"], ate["score"], ate["nuis"], sgm["seg0"], config.weight_cap, "png/ate")
dr_cdf = oc_dr_cdf_by_seg(prcs["A"], prcs["Y"], ate["nuis"], sgm["seg0"], None, prcs["level"])
drcdf_plot(cdf_seg=dr_cdf, output_dir="png/drcdf", show=True)
hei = hei_result(ate["nuis"], prcs["A"], prcs["Y"], prcs["S"] ,sgm["per_seg"], prcs["score"])

print("X shape:", prcs["X"].shape)
print("features:", prcs["ftr"])
print("A counts:")
print(pd.Series(prcs["A"]).value_counts().sort_index())
print("Y counts:")
print(pd.Series(prcs["Y"]).value_counts().sort_index())
print("S counts:")
print(pd.Series(prcs["S"]).value_counts().sort_index())