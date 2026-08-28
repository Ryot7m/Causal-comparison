import numpy as np

from workspace.aipw import (
    fit_cf_oc,
    oc_aipw_ate,
)


def test_oc_aipw_recovers_known_effect():
    # 結果が処置変数と完全に一致するデータ
    treatment = np.array([0, 1, 0, 1])
    outcome = np.array([0, 1, 0, 1])
    segment = np.zeros(4, dtype=int)
    score_values = np.array([0.0, 1.0])

    # 真の条件付き平均を与える
    nuisance = {
        "e_hat": np.full(4, 0.5),
        "mu1_hat": np.ones(4),
        "mu0_hat": np.zeros(4),
    }

    result = oc_aipw_ate(
        A=treatment,
        Y=outcome,
        s_values=score_values,
        nuisance=nuisance,
        seg=segment,
        cap=None,
    )

    expected_columns = {
        "cls",
        "clsnum",
        "ate",
        "se",
        "ci_low",
        "ci_high",
    }

    assert set(result.columns) == expected_columns
    assert len(result) == 1

    row = result.iloc[0]

    assert row["cls"] == 0
    assert row["clsnum"] == 4
    assert np.isclose(row["ate"], 1.0)
    assert np.isclose(row["se"], 0.0)
    assert np.isclose(row["ci_low"], 1.0)
    assert np.isclose(row["ci_high"], 1.0)


def test_fit_cf_oc_returns_valid_nuisance_predictions():
    n = 30
    index = np.arange(n)

    features = np.column_stack([
        index / n,
        np.sin(index),
    ])
    treatment = (index % 2).astype(int)
    outcome = (index % 3).astype(int)
    score_values = np.array([1.0, 2.0, 3.0])

    result = fit_cf_oc(
        X=features,
        A=treatment,
        Y=outcome,
        s_values=score_values,
        n_splits=2,
        random_state=123,
    )

    assert result["e_hat"].shape == (n,)
    assert result["p_hat1"].shape == (n, 3)
    assert result["p_hat0"].shape == (n, 3)
    assert result["mu1_hat"].shape == (n,)
    assert result["mu0_hat"].shape == (n,)
    assert result["folds"].shape == (n,)

    assert ((result["e_hat"] > 0) & (result["e_hat"] < 1)).all()

    np.testing.assert_allclose(
        result["p_hat1"].sum(axis=1),
        1.0,
    )
    np.testing.assert_allclose(
        result["p_hat0"].sum(axis=1),
        1.0,
    )

    assert np.isfinite(result["mu1_hat"]).all()
    assert np.isfinite(result["mu0_hat"]).all()

    np.testing.assert_allclose(
        result["mu1_hat"],
        result["p_hat1"] @ score_values,
    )
    np.testing.assert_allclose(
        result["mu0_hat"],
        result["p_hat0"] @ score_values,
    )