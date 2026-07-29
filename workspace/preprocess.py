import pandas as pd
import numpy as np

def pre_analysis(data, config):

    data["Feature_1"] = 6 - data["Feature_1"]
    # data = data.fillna(0)
    data_filt = data.filter(regex="Q16", axis = 1)
    data_filt = 5 - data_filt
    data = pd.concat([data, data_filt], axis = 1)

    # 列名の変更(列名に応じて変更をしなくても可)
    data = data.rename(columns={
        "満足度A":"Feature_1",
        "満足度B":"Feature_2",
        "満足度C":"Feature_12",
        "期待値":"except",
        "処置":"Treatment",
        "アウトカム変数":"Outcome"
    })

    outcome_col = "Outcome"
    for col, max_value in config.reverse_score_max.items():
        data[col] = max_value - data[col]

    required = [
        config.treatment_col,
        config.outcome_col,
        config.segment_col,
        *config.confounder_cols,
    ]
    data = data.dropna(subset=required).copy()

    y_to_index = {
        level: i for i, level in enumerate(config.outcome_levels)
    }
    state_col = "Feature_1"
    score_rec = np.array([0, 1, 2, 3, 4],dtype= "float") 

    th_sh = data[state_col].quantile(0.75)
    data["Treatment"] = (data[state_col] >= th_sh).astype(int)
    seg_col = "except"

    #交絡因子
    ftr_cols = [c for c in data.columns]
    ftr_cols.remove("Outcome")
    ftr_cols = [i for i in ftr_cols if i not in ['Feature_1', 'Feature_2', 'Feature_3',"Feature_4"]]
    ftr_cols.remove("Treatment") #必要に応じて変数の削除
    ftr_cols.remove(seg_col)

    X = pd.get_dummies(data[ftr_cols], drop_first=False).values
    A = data["Treatment"].astype(int).values

    levels = pd.Series(data[outcome_col]).dropna().unique().tolist()
    levels_sorted = sorted(levels)  
    # y_to_index = {lev:i for i, lev in enumerate(levels_sorted)}

    outcome_levels = [1, 2, 3, 4, 5]  # 必ず意味上の順序で指定

    y_to_index = {level: i for i, level in enumerate(outcome_levels)}
    Y = data["Outcome"].map(y_to_index).astype(int).to_numpy()
    # Y = pd.Series(data[outcome_col]).map(y_to_index).astype(int).values

    K = len(levels_sorted)
    
    return {
        "seg" : seg_col,
        "ftr" : ftr_cols,
        "level" : levels_sorted,
        "X" : X,
        "A" : A,
        "Y" : Y,
        "cap" : np.asarray(config.score_values)
    }