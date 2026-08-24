import numpy as np
import pandas as pd
import pytest

from app.dantic import (
    BinaryColumnTreatment,
    QuantileTreatment,
)
from workspace.preprocess import create_treatment


@pytest.mark.parametrize(
    (
        "values",
        "treated_value",
        "control_value",
        "expected",
    ),
    [
        (
            [0, 1, 0, 1],
            1,
            0,
            [0, 1, 0, 1],
        ),
        (
            ["no", "yes", "no", "yes"],
            "yes",
            "no",
            [0, 1, 0, 1],
        ),
    ],
)
def test_create_treatment_maps_binary_values(
    values,
    treated_value,
    control_value,
    expected,
):
    data = pd.DataFrame({
        "treatment": values,
    })

    treatment = BinaryColumnTreatment(
        mode="binary_column",
        column="treatment",
        treated_value=treated_value,
        control_value=control_value,
    )

    result = create_treatment(
        data,
        treatment,
    )

    np.testing.assert_array_equal(
        result.values.to_numpy(),
        np.array(expected),
    )

    assert result.threshold is None


def test_create_treatment_rejects_unknown_binary_value():
    data = pd.DataFrame({
        "treatment": [0, 1, 2],
    })

    treatment = BinaryColumnTreatment(
        mode="binary_column",
        column="treatment",
        treated_value=1,
        control_value=0,
    )

    with pytest.raises(
        ValueError,
        match="未定義の値",
    ):
        create_treatment(
            data,
            treatment,
        )


@pytest.mark.parametrize(
    (
        "treated_when",
        "expected",
    ),
    [
        ("ge", [False, True, True]),
        ("gt", [False, False, True]),
        ("le", [True, True, False]),
        ("lt", [True, False, False]),
    ],
)
def test_create_treatment_supports_quantile_comparisons(
    treated_when,
    expected,
):
    data = pd.DataFrame({
        "source": [1.0, 2.0, 3.0],
    })

    treatment = QuantileTreatment(
        mode="quantile",
        source_column="source",
        quantile=0.5,
        treated_when=treated_when,
    )

    result = create_treatment(
        data,
        treatment,
    )

    assert result.threshold == 2.0

    np.testing.assert_array_equal(
        result.values.to_numpy(),
        np.array(expected),
    )


def test_create_treatment_rejects_non_numeric_quantile_source():
    data = pd.DataFrame({
        "source": ["low", "middle", "high"],
    })

    treatment = QuantileTreatment(
        mode="quantile",
        source_column="source",
        quantile=0.5,
    )

    with pytest.raises(
        ValueError,
        match="数値列",
    ):
        create_treatment(
            data,
            treatment,
        )


def test_create_treatment_rejects_all_missing_quantile_source():
    data = pd.DataFrame({
        "source": [np.nan, np.nan, np.nan],
    })

    treatment = QuantileTreatment(
        mode="quantile",
        source_column="source",
        quantile=0.5,
    )

    with pytest.raises(
        ValueError,
        match="有効な数値",
    ):
        create_treatment(
            data,
            treatment,
        )


def test_create_treatment_preserves_missing_source():
    data = pd.DataFrame({
        "source": [1.0, np.nan, 3.0],
    })

    treatment = QuantileTreatment(
        mode="quantile",
        source_column="source",
        quantile=0.5,
        treated_when="ge",
    )

    result = create_treatment(
        data,
        treatment,
    )

    assert result.values.iloc[0] == 0
    assert pd.isna(result.values.iloc[1])
    assert result.values.iloc[2] == 1