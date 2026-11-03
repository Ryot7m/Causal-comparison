import math

import numpy as np
import pandas as pd
import pytest
import copy

from app.database import SessionLocal
from app.models import AnalysisJob
from app.renew import create_analysis_job
from app.tasks import run_analysis_job


@pytest.mark.slow
@pytest.mark.parametrize(
    ("seed_input", "expected_seed"),
    [(None, 123), (42, 42)],
    ids=["default-123", "explicit-42"],
)
@pytest.mark.parametrize(
    "outcome_levels",
    [
        [1, 2, 3, 4, 5],
        ["very_low", "low", "middle", "high", "very_high"],
    ],
)
def test_analysis_job_runs_full_pipeline(config, outcome_levels, seed_input, expected_seed, tmp_path):
    rng = np.random.default_rng(42)

    rows_per_segment = 500
    total_rows = rows_per_segment * 3
    within_segment = np.tile(np.arange(rows_per_segment), 3)

    outcome_labels = np.asarray(outcome_levels, dtype=object)

    data = pd.DataFrame({
        "Q4_1": outcome_labels[within_segment % 5],
        "Q7_4": np.repeat(
            [1.0, 2.0, 3.0],
            rows_per_segment,
        ),
        "Q2_9": within_segment.astype(float),
        "x1": rng.normal(size=total_rows),
        "x2": rng.normal(size=total_rows),
    })
    
    request_config = copy.deepcopy(config)

    if seed_input is None:
        request_config.pop("split_random_state", None)
    else:
        request_config["split_random_state"] = seed_input

    request_config["outcome"]["levels"] = outcome_levels
    request_config["outcome"]["scores"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    request_config["covariates"]["columns"] = ["x1", "x2"]
    
    csv_path = tmp_path / "valid.csv"
    data.to_csv(csv_path, index=False)

    with SessionLocal() as db:
        job = create_analysis_job(
            db=db,
            config=request_config,
            filename="valid.csv",
            input_path=str(csv_path),
        )
        job_id = job.id

    # Redisを通さず、Celeryタスク本体をテスト内で直接実行
    run_analysis_job.run(str(job_id))

    with SessionLocal() as db:
        job = db.get(AnalysisJob, job_id)

        assert job is not None
        assert job.status == "succeeded", job.error_message
        assert job.result is not None
        body = job.result

    assert set(body) == {
        "segment",
        "ate",
        "drcdf",
        "hei",
        "split_random_state"
    }

    assert body["segment"] == {
        "cut1": 1.0,
        "cut2": 2.0
    }

    # 3セグメント分のATE
    assert len(body["ate"]) == 3
    assert {
        row["cls"]
        for row in body["ate"]
    } == {0, 1, 2}

    assert all(row["clsnum"] > 0 for row in body["ate"])
    assert sum(row["clsnum"] for row in body["ate"]) == total_rows // 2

    assert all(
        math.isfinite(row["ate"])
        and math.isfinite(row["se"])
        and math.isfinite(row["ci_low"])
        and math.isfinite(row["ci_high"])
        for row in body["ate"]
    )
    
    assert {row["threshold"] for row in body["drcdf"]} == set(outcome_levels)
    
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
