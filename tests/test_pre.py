import numpy as np
import pandas as pd
import pytest

from workspace.preprocess import pre_analysis


def test_pre_analysis_creates_expected_treatment(analysis_data, analysis_config):
    result = pre_analysis(analysis_data, analysis_config)

    # stateの中央値は2.5なので、3と4が処置群
    np.testing.assert_array_equal(
        result["A"],
        np.array([0, 0, 1, 1]),
    )

    assert len(result["X"]) == 4
    assert len(result["Y"]) == 4
    assert len(result["S"]) == 4


def test_pre_analysis_zero_fills_missing_value(analysis_data, analysis_config):
    analysis_data.loc[0, "x1"] = np.nan

    result = pre_analysis(
        analysis_data,
        analysis_config
    )

    assert result["X"][0, 0] == 0


def test_pre_analysis_rejects_unknown_outcome(analysis_data, analysis_config):
    analysis_data.loc[0, "outcome"] = 99

    with pytest.raises(
        ValueError,
        match="outcome_levels にない値",
    ):
        pre_analysis(
            analysis_data,
            analysis_config
        )
        
def test_pre_analysis_rejects_single_treatment_group(analysis_data, analysis_config):
    # segment=2の行は処置群なので、除外すると対照群だけが残る
    analysis_config.segment_missing_values = (2,)

    with pytest.raises(
        ValueError,
        match="処置群（1）と対照群（0）の両方",
    ):
        pre_analysis(
            analysis_data,
            analysis_config
        )
        
def test_pre_analysis_excludes_segment_missing_before_outcome_levels(analysis_config):
    data = pd.DataFrame({
        "outcome": [1, 2, 1, 2, 1, 3],
        "segment": [1, 1, 1, 1, 1, -1],
        "state": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    })

    analysis_config.segment_missing_values = (-1,)

    result = pre_analysis(data, analysis_config)

    assert result["level"] == [1, 2]

    np.testing.assert_array_equal(
        result["score"],
        np.array([1.0, 2.0])
    )

    assert set(result["Y"]) == {0, 1}
    
def test_pre_analysis_drop_removes_missing_confounder_row(analysis_data, analysis_config):
    analysis_config.missing_type = "drop"
    analysis_data.loc[0, "x1"] = np.nan

    result = pre_analysis(
        analysis_data,
        analysis_config
    )

    assert len(result["X"]) == 3
    assert len(result["A"]) == 3
    assert len(result["Y"]) == 3
    assert not np.isnan(result["X"]).any()
    
def test_pre_analysis_fill_uses_configured_value(analysis_data, analysis_config):
    analysis_config.missing_type = "fill"
    analysis_config.fill_values = {
        "x1": -999.0
    }
    analysis_data.loc[0, "x1"] = np.nan

    result = pre_analysis(analysis_data, analysis_config)

    assert len(result["X"]) == 4
    assert result["X"][0, 0] == -999.0
    
def test_pre_analysis_rejects_fill_for_non_covariate(analysis_data, analysis_config):
    analysis_config.missing_type = "fill"
    analysis_config.fill_values = {
        "outcome": 0,
    }

    with pytest.raises(ValueError,match="交絡変数だけ"):
        pre_analysis(analysis_data, analysis_config)

def test_pre_analysis_zero_does_not_fill_outcome(analysis_data, analysis_config):
    analysis_config.missing_type = "zero"
    analysis_data.loc[0, "outcome"] = np.nan

    result = pre_analysis(analysis_data, analysis_config)

    # outcome欠損行は0補完せず除外される
    assert len(result["Y"]) == 3
    assert len(result["X"]) == 3
    
def test_pre_analysis_fills_categorical_covariate_before_encoding(analysis_data, analysis_config):
    analysis_data["category"] = [
        "A",
        None,
        "B",
        "A"
    ]

    analysis_config.confounder_cols = [
        "x1",
        "category"
    ]
    analysis_config.categorical_cols = [
        "category"
    ]
    analysis_config.missing_type = "fill"
    analysis_config.fill_values = {
        "category": "missing"
    }

    result = pre_analysis(
        analysis_data,
        analysis_config
    )

    assert len(result["X"]) == 4
    assert "category_missing" in result["ftr"]

    missing_index = result["ftr"].index(
        "category_missing"
    )

    assert result["X"][1, missing_index] == 1