"""
このコードは研究の流れを基に各関数を呼び出して、
それぞれの機能を繋げる役割を持つコードである
"""

import numpy as np
import pandas as pd

# import japanize_matplotlib
from workspace.segmentation import segmentation_rtn
from workspace.aipw import aipw_ate
from workspace.ateplot import ate_plot
from workspace.drcdf import drcdf_plot
from workspace.hei import hei_result 
from workspace.preprocess import pre_analysis

data = pd.read_csv("sample.csv", encoding="shift-jis")


config = {
    "treatment_col": "広告接触",
    "treatment_positive_value": "接触あり",
    "outcome_col": "推奨意向",
    "segment_col": "事前期待",
    "confounder_cols": ["年齢", "性別", "利用頻度", "事前満足度"],
    "outcome_levels": [1, 2, 3, 4, 5],
    "score_values": [1, 2, 3, 4, 5],
    "reverse_score_max": {"Q16_1": 5, "Q16_2": 5},
    "weight_cap": 100,
}

prcs = pre_analysis() 
sgm = segmentation_rtn(data, seg_col, ftr_cols, A, X, Y)
ate = aipw_ate(sgm["X0"], sgm["A0"], sgm["Y0"], sgm["seg0"], 100)
ate_plot(sgm["A0"], sgm["Y0"], ate["score"], ate["nuis"], sgm["seg0"])
drcdf_plot(sgm["A0"], sgm["Y0"], ate["nuis"], sgm["seg0"], levels_sorted)
hei_result(ate["nuis"], sgm["A0"], sgm["Y0"], sgm["S0"] ,sgm["per_seg"])