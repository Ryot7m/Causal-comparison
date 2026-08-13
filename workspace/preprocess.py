import pandas as pd
import numpy as np
from fnmatch import fnmatch
from dataclasses import dataclass

@dataclass(frozen=True)
class TreatmentResult:
    values: pd.Series
    threshold: float | None = None

def create_treatment(data, treatment):
    if treatment.mode == "binary_column":
        source = data[treatment.column]

        allowed = {
            treatment.treated_value,
            treatment.control_value,
        }

        unknown = set(source.dropna().unique()) - allowed
        if unknown:
            raise ValueError(
                f"処置列に未定義の値があります: {sorted(unknown)}"
            )

        return TreatmentResult(values=source.map({
            treatment.control_value: 0,
            treatment.treated_value: 1,
        }))

    if treatment.mode == "quantile":
        values = (source >= treatment.threshold).where(source.notna())

    return TreatmentResult(values=values, threshold=float(treatment.threshold))

def pre_analysis(data_, config):
    data = data_.copy()
    
    treatment_result = create_treatment(
        data, config.treatment,
    )

    data[config.treatment_col] = (
        treatment_result.values
    )
    
    for col, max_value in config.reverse_score_max.items():
        data[col] = max_value - data[col]
    
    if config.confounder_cols is None:
        other = {
            config.treatment_col,
            config.outcome_col,
            config.segment_col,
            config.state_col,
            *config.exclude_cols,
        }

        if config.treatment_source_col is not None:
            other.add(config.state_col)

        confounder = [
            col for col in data.columns
            if col not in other]
    else:
        confounder = list(config.confounder_cols)
        
    confounder = [
    col for col in confounder
    if not any(
        fnmatch(col, pattern)
        for pattern in config.exclude_conditions
    )]
        
    if config.missing_type == "zero":
    # Noneなら全共変量を0埋め
        if config.zero_fill is None:
            fill_cols = list(confounder)
        else:
            fill_cols = list(config.zero_fill)

        unknown = set(fill_cols) - set(data.columns)
        if unknown:
            raise ValueError(
                f"0埋め対象に存在しない列があります: {sorted(unknown)}"
            )

        # 0埋めしない変数の設定（処置・アウトカム・セグメント）
        omit = {
            config.treatment_col,
            config.outcome_col,
            config.segment_col,
            config.state_col,
        }
        fill_cols = [col for col in fill_cols if col not in omit]

        data[fill_cols] = data[fill_cols].fillna(0)

    elif config.missing_type != "drop":
        raise ValueError(
            "missing_strategy は 'drop' または 'zero' を指定してください。"
        )

    required = [
        config.treatment_col,
        config.outcome_col,
        config.segment_col,
        *confounder,
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

    y_to_index = {level: i for i, level in enumerate(levels)}
    Y = data[config.outcome_col].map(y_to_index).astype(int).to_numpy()
    # Y = pd.Series(data[outcome_col]).map(y_to_index).astype(int).values
    S = data[config.segment_col].replace(list(config.segment_missing_values), np.nan).to_numpy()

    # K = len(levels_sorted)
    
    ok = ~pd.isna(S)
    X_df0 = pd.get_dummies(data.loc[ok, confounder], drop_first=False)
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
        "score" : score,
        "treatment_threshold": treatment_result.threshold
    }