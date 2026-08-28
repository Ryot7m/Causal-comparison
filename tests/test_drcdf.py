import numpy as np

from workspace.drcdf import oc_dr_cdf_by_seg


def test_drcdf_returns_expected_structure():
    treatment = np.array([0, 0, 1, 1])
    outcome = np.array([0, 1, 0, 1])
    segment = np.zeros(4, dtype=int)

    nuisance = {
        "e_hat": np.full(4, 0.5),
        "p_hat1": np.full((4, 2), 0.5),
        "p_hat0": np.full((4, 2), 0.5),
    }

    result = oc_dr_cdf_by_seg(
        A=treatment,
        Y=outcome,
        nuis=nuisance,
        seg=segment,
        cap=None,
        level_labels=[1, 2],
    )

    expected_columns = {
        "seg",
        "c",
        "threshold",
        "F1_dr",
        "F0_dr",
        "tau_c",
        "se_c",
        "ci_low",
        "ci_high",
    }

    assert set(result.columns) == expected_columns
    assert len(result) == 2

    assert result["F1_dr"].between(0, 1).all()
    assert result["F0_dr"].between(0, 1).all()
    assert (result["se_c"] >= 0).all()

    np.testing.assert_allclose(
        result["tau_c"],
        result["F1_dr"] - result["F0_dr"],
    )


def test_drcdf_last_threshold_is_one():
    treatment = np.array([0, 0, 1, 1])
    outcome = np.array([0, 1, 0, 1])
    segment = np.zeros(4, dtype=int)

    nuisance = {
        "e_hat": np.full(4, 0.5),
        "p_hat1": np.full((4, 2), 0.5),
        "p_hat0": np.full((4, 2), 0.5),
    }

    result = oc_dr_cdf_by_seg(
        A=treatment,
        Y=outcome,
        nuis=nuisance,
        seg=segment,
        level_labels=[1, 2],
    )

    last = result.sort_values("c").iloc[-1]

    assert np.isclose(last["F1_dr"], 1.0)
    assert np.isclose(last["F0_dr"], 1.0)
    assert np.isclose(last["tau_c"], 0.0)