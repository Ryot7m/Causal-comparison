import pandas as pd
import numpy as np

def pre_analysis(data_, config):
    data = data_.copy()
    
    for col, max_value in config.reverse_score_max.items():
        data[col] = max_value - data[col]
        
    source = data[config.state_col]
    th_sh = source.quantile(config.threshold)

    # 欠損値を対照群(0)に誤分類しない
    data[config.treatment_col] = (source >= th_sh).where(source.notna())

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

    levels_all = list(config.outcome_levels)
    score_all = np.asarray(config.score_values, dtype=float)

    if len(levels_all) != len(score_all):
        raise ValueError("outcome_levels と score_values の長さが一致していません。")

    observe_set = set(data[config.outcome_col].dropna())
    nothing = observe_set - set(levels_all)
    if nothing:
        raise ValueError(f"outcome_levels にない値があります: {sorted(nothing)}")

    keep = [i for i, level in enumerate(levels_all) if level in observe_set]
    levels = [levels_all[i] for i in keep]
    score = score_all[keep]

    if len(levels) < 2:
        raise ValueError("分析には少なくとも2つのアウトカムが必要です。")

    y_to_index = {level: i for i, level in enumerate(config.outcome_levels)}
    Y = data[config.outcome_col].map(y_to_index).astype(int).to_numpy()
    # Y = pd.Series(data[outcome_col]).map(y_to_index).astype(int).values
    S = data[config.segment_col].replace(list(config.segment_missing_values), np.nan).to_numpy()

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
        "level" : levels,
        "X" : X0,
        "A" : A0,
        "Y" : Y0,
        "S" : S0,
        "treat" : A,
        "ftr" : feature_names,
        "score" : score
    }