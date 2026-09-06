import pandas as pd
import numpy as np
from app.dantic import QuantileTreatment
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
            treatment.control_value
        }

        unknown = set(source.dropna().unique()) - allowed
        if unknown:
            raise ValueError(
                f"処置列に未定義の値があります: {sorted(unknown)}"
            )

        return TreatmentResult(values=source.map({
            treatment.control_value: 0,
            treatment.treated_value: 1
        }))

    if treatment.mode == "quantile":
        source = data[treatment.source_column]

        if not pd.api.types.is_numeric_dtype(source):
            raise ValueError(
                "分位点処置の生成元には"
                "数値列を指定してください。"
            )

        threshold = float(
            source.quantile(treatment.quantile)
        )
        
        if not np.isfinite(threshold):
            raise ValueError(
                "分位点処置の生成元に"
                "有効な数値がありません。")

        comparisons = {
            "ge": source.ge,
            "gt": source.gt,
            "le": source.le,
            "lt": source.lt
        }

        values = comparisons[
            treatment.treated_when
        ](threshold).where(source.notna())

        return TreatmentResult(
            values=values,
            threshold=threshold,
        )

    raise ValueError(
        f"未対応の処置方式です: {treatment.mode}"
    )
    
def pre_analysis(data_, config):
    data = data_.copy()
    
    treatment = QuantileTreatment(
        mode="quantile",
        source_column="Q2_10",
        quantile=0.5,
        treated_when="ge"
        )
    
    treatment_result = create_treatment(
        data, treatment
    )

    data[config.treatment_col] = (
        treatment_result.values
    )

    confounder = list(config.confounder_cols)
        
    if config.missing_type == "zero":
        data[confounder] = data[confounder].fillna(0)
    
    elif config.missing_type == "fill":
        fill_cols = list(config.fill_values)

        unknown = set(fill_cols) - set(confounder)
        if unknown:
            raise ValueError(
                "fill_valuesには交絡変数だけを指定してください: "
                f"{sorted(unknown)}"
            )

        data[fill_cols] = data[fill_cols].fillna(
            value=config.fill_values
        )

    elif config.missing_type != "drop":
        raise ValueError(
            "missing strategyは"
            "'drop','fill','zero'のいずれかを指定してください。"
        )
    
    for col in config.categorical_cols:
        data[col] = data[col].astype("category")
        
    data[config.segment_col] = data[
        config.segment_col
        ].replace(list(config.segment_missing_values),
                  np.nan)

    required = [
        config.treatment_col,
        config.outcome_col,
        config.segment_col,
        *confounder
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
    
    groups, counts = np.unique(A0, return_counts=True)

    group_counts = {
        int(group): int(count)
        for group, count in zip(groups, counts)
    }

    if set(group_counts) != {0, 1}:
        raise ValueError(
            "分析対象データには処置群（1）と"
            "対照群（0）の両方が必要です。"
            f"群別件数: {group_counts}"
        )
    
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