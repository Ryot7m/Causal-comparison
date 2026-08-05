import numpy as np

from workspace.segmentation import (
    ess,
    make_seg_from_cuts,
    weighted_smd,
)


def test_make_seg_from_cuts():
    values = np.array([0, 1, 2, 3, 4])

    result = make_seg_from_cuts(
        values,
        cut1=1,
        cut2=3,
    )

    np.testing.assert_array_equal(
        result,
        np.array([0, 0, 1, 1, 2]),
    )


def test_ess_with_equal_weights_equals_sample_size():
    weights = np.ones(4)

    result = ess(weights)

    assert np.isclose(result, 4.0)


def test_weighted_smd_is_zero_for_identical_groups():
    features = np.array([
        [0.0],
        [1.0],
        [0.0],
        [1.0],
    ])

    treatment = np.array([1, 1, 0, 0])
    propensity = np.full(4, 0.5)

    max_smd, mean_smd = weighted_smd(
        features,
        treatment,
        propensity,
    )

    assert np.isclose(max_smd, 0.0)
    assert np.isclose(mean_smd, 0.0)