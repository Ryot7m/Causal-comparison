import numpy as np

from workspace.hei import (
    calculate_heterogeneity_score,
)


def test_heterogeneity_is_zero_for_constant_outcome():
    pseudo_outcomes = np.ones(100)
    predicted_scores = np.arange(100)

    result = calculate_heterogeneity_score(
        pseudo_outcomes,
        predicted_scores,
    )

    assert np.isclose(result, 0.0)