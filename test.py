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
from workspace.drcdf import drcdf_plot
from workspace.hei import hei_result 
from workspace.preprocess import pre_analysis

data = pd.read_csv("sample.csv", encoding="shift-jis")

@dataclass
class AnalysisConfig:
    treatment_col: str
    outcome_col: str
    segment_col: str
    confounder_cols: list[str]

    outcome_levels: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    score_values: list[float] = field(
        default_factory=lambda: [1, 2, 3, 4, 5]
    )
    reverse_score_max: dict[str, float] = field(default_factory=dict)
    weight_cap: 100


config = AnalysisConfig(
    treatment_col="Treatment",
    outcome_col="Outcome",
    segment_col="except",
    confounder_cols=["SQ1", "SQ3", "SQ8", "Q5_2"],
    reverse_score_max={"Feature_1": 6},
)

prcs = pre_analysis(data, config) 
sgm = segmentation_rtn(prcs["S"], prcs["seg"], prcs["ftr"], prcs["A"], prcs["X"], prcs["Y"], prcs["treat"])
ate = aipw_ate(sgm["X0"], sgm["A0"], sgm["Y0"], sgm["seg0"], 100)
ate_plot(sgm["A0"], sgm["Y0"], ate["score"], ate["nuis"], sgm["seg0"])
drcdf_plot(sgm["A0"], sgm["Y0"], ate["nuis"], sgm["seg0"], prcs["level"])
hei_result(ate["nuis"], sgm["A0"], sgm["Y0"], sgm["S0"] ,sgm["per_seg"])