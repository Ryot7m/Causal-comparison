import numpy as np
import pandas as pd

from workspace.aipw import fit_cf_oc, oc_aipw_ate


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def make_synthetic_data(
    n: int = 10_000,
    seed: int = 2026,
    effect: float = 0.0,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)

    age = rng.normal(size=n)
    region = rng.choice(["A", "B"], size=n)
    region_b = (region == "B").astype(float)

    # ageとregionの両方がtreatmentとoutcomeに影響する
    propensity = 1 / (
        1 + np.exp(-(-0.2 + 0.4 * age + 0.3 * region_b))
    )
    treatment = rng.binomial(1, propensity)

    def outcome_probabilities(a: int) -> np.ndarray:
        logits = np.column_stack([
            np.zeros(n),
            -0.3 + 0.25 * age + 0.2 * region_b
            + a * 0.5 * effect,
            -0.8 + 0.4 * age - 0.2 * region_b
            + a * effect,
        ])
        return _softmax(logits)

    # 各行について、処置あり・なし両方の確率を計算する
    prob0 = outcome_probabilities(a=0)
    prob1 = outcome_probabilities(a=1)

    # 共通の一様乱数を使ってカテゴリoutcomeを生成する
    u = rng.uniform(size=n)
    observed_prob = np.where(
        treatment[:, None] == 1,
        prob1,
        prob0,
    )
    category_index = (
        u[:, None] > np.cumsum(observed_prob, axis=1)
    ).sum(axis=1)

    levels = np.array(["low", "middle", "high"])
    outcome = levels[category_index]

    data = pd.DataFrame({
        "treatment": treatment,
        "outcome": outcome,
        "age": age,
        "region": region,
        # セグメント選択用。今回は効果や処置割当には影響させない
        "segment_score": rng.uniform(size=n),
    })

    scores = np.array([1.0, 2.0, 3.0])
    expected_y0 = prob0 @ scores
    expected_y1 = prob1 @ scores

    # このデータにおける、生成モデル上の正解ATE
    true_ate = float(np.mean(expected_y1 - expected_y0))

    return data, true_ate

ATE_TOLERANCE = 0.1  # 仮の値。推定誤差を確認して調整する


def test_cf_oc_aipw_recovers_known_effect():
    data, true_ate = make_synthetic_data(
        n=10_000,
        seed=2026,
        effect=0.8,
    )

    X = pd.get_dummies(
        data[["age", "region"]],
        columns=["region"],
        dtype=float,
    ).to_numpy(dtype=float)

    A = data["treatment"].to_numpy(dtype=int)
    Y = pd.Categorical(
        data["outcome"],
        categories=["low", "middle", "high"],
        ordered=True,
    ).codes

    score_values = np.array([1.0, 2.0, 3.0])

    nuisance = fit_cf_oc(
        X=X,
        A=A,
        Y=Y,
        s_values=score_values,
        n_splits=5,
        random_state=2026,
    )

    result = oc_aipw_ate(
        A=A,
        Y=Y,
        s_values=score_values,
        nuisance=nuisance,
        seg=np.zeros(len(data), dtype=int),
        cap=None,
    )

    row = result.iloc[0]
    estimated_ate = float(row["ate"])

    assert estimated_ate > 0
    assert abs(estimated_ate - true_ate) < ATE_TOLERANCE