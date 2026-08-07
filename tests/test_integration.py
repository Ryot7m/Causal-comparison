import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.slow
def test_estimate_endpoint_runs_full_pipeline():
    rng = np.random.default_rng(42)

    rows_per_segment = 120
    total_rows = rows_per_segment * 3
    within_segment = np.tile(
        np.arange(rows_per_segment),
        3,
    )

    data = pd.DataFrame({
        "Q4_1": (within_segment % 5) + 1,
        "Q7_4": np.repeat(
            [1.0, 2.0, 3.0],
            rows_per_segment,
        ),
        "Q2_9": within_segment.astype(float),
        "x1": rng.normal(size=total_rows),
        "x2": rng.normal(size=total_rows),
    })

    csv = data.to_csv(
        index=False,
    ).encode("utf-8")

    with TestClient(app) as client:
        response = client.post(
            "/api/estimate",
            files={
                "file": (
                    "synthetic.csv",
                    csv,
                    "text/csv",
                ),
            },
        )

    assert response.status_code == 200, response.text

    body = response.json()

    assert set(body) == {
        "segment",
        "ate",
        "drcdf",
        "hei",
    }

    assert body["segment"] == {
        "cut1": 1.0,
        "cut2": 2.0,
    }

    # 3セグメント分のATE
    assert len(body["ate"]) == 3
    assert {
        row["cls"]
        for row in body["ate"]
    } == {0, 1, 2}

    assert all(
        row["clsnum"] == 120
        for row in body["ate"]
    )

    assert all(
        math.isfinite(row["ate"])
        and math.isfinite(row["se"])
        and math.isfinite(row["ci_low"])
        and math.isfinite(row["ci_high"])
        for row in body["ate"]
    )

    # 3セグメント × 5アウトカム水準
    assert len(body["drcdf"]) == 15

    assert {
        row["seg"]
        for row in body["drcdf"]
    } == {0, 1, 2}

    assert {
        row["c"]
        for row in body["drcdf"]
    } == {0, 1, 2, 3, 4}

    assert all(
        0 <= row["F1_dr"] <= 1
        and 0 <= row["F0_dr"] <= 1
        and row["se_c"] >= 0
        for row in body["drcdf"]
    )

    assert math.isfinite(
        body["hei"]["score"]
    )