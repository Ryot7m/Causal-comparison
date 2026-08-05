import numpy as np
import pytest

from workspace.preprocess import pre_analysis


def test_pre_analysis_creates_expected_treatment(
    analysis_data,
    analysis_config,
):
    result = pre_analysis(
        analysis_data,
        analysis_config,
    )

    # stateの中央値は2.5なので、3と4が処置群
    np.testing.assert_array_equal(
        result["A"],
        np.array([0, 0, 1, 1]),
    )

    assert len(result["X"]) == 4
    assert len(result["Y"]) == 4
    assert len(result["S"]) == 4


def test_pre_analysis_zero_fills_missing_value(
    analysis_data,
    analysis_config,
):
    analysis_data.loc[0, "x1"] = np.nan

    result = pre_analysis(
        analysis_data,
        analysis_config,
    )

    assert result["X"][0, 0] == 0


def test_pre_analysis_rejects_unknown_outcome(
    analysis_data,
    analysis_config,
):
    analysis_data.loc[0, "outcome"] = 99

    with pytest.raises(
        ValueError,
        match="outcome_levels にない値",
    ):
        pre_analysis(
            analysis_data,
            analysis_config,
        )