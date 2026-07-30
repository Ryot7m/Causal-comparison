import pandas as pd
import numpy as np

def pre_analysis(data, config):
    
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
    
    # th_sh = data[config.reverse_score_max].quantile(0.75)
    # data["Treatment"] = (data[config.reverse_score_max] >= th_sh).astype(int)

    # X = pd.get_dummies(data[ftr_cols], drop_first=False).values
    A = data[config.treatment_col].astype(int).values

    levels = pd.Series(data[config.treatment_col]).dropna().unique().tolist()
    levels_sorted = sorted(levels)  
    # y_to_index = {lev:i for i, lev in enumerate(levels_sorted)}

    outcome_levels = [1, 2, 3, 4, 5]  # 必ず意味上の順序で指定

    y_to_index = {level: i for i, level in enumerate(outcome_levels)}
    Y = data["Outcome"].map(y_to_index).astype(int).to_numpy()
    # Y = pd.Series(data[outcome_col]).map(y_to_index).astype(int).values
    S = data[config.segment_col].replace(0, np.nan).to_numpy()

    # K = len(levels_sorted)
    
    ok = ~pd.isna(S)
    X_df0 = pd.get_dummies(data.loc[ok, config.confounder_cols], drop_first=False)
    feature_names = list(X_df0.columns)
    X0 = X_df0.values
    
    S0 = S[ok]
    A0 = A[ok]
    Y0 = Y[ok].astype(int)
    
    return {
        "seg" : config.segment_col,
        "level" : levels_sorted,
        "X" : X0,
        "A" : A0,
        "Y" : Y0,
        "S" : S0,
        "treat" : A,
        "ftr" : feature_names,
        "score" : np.asarray(config.score_values, dtype=float)
    }