import json

from fastapi.testclient import TestClient
from app.main import app
from uuid import uuid4, UUID
from pathlib import Path
from types import SimpleNamespace
import app.api as api
from app.database import SessionLocal
from app.renew import create_analysis_job, mark_analysis_succeeded


client = TestClient(app)

def test_estimate_accepts_valid_csv(monkeypatch, config, tmp_path):
    captured = {}
    expected_job_id = UUID("11111111-1111-4111-8111-111111111111")

    def fake_csv_save(contents):
        path = tmp_path / "uploaded.csv"
        path.write_bytes(contents)
        captured["saved_path"] = path
        return str(path)

    def fake_create_analysis_job(db, config, filename, input_path):
        captured["filename"] = filename
        captured["input_path"] = input_path
        return SimpleNamespace(id=expected_job_id)

    class FakeTask:
        def delay(self, job_id):
            captured["queued_job_id"] = job_id

    monkeypatch.setattr(api, "csv_save", fake_csv_save)
    monkeypatch.setattr(api, "create_analysis_job", fake_create_analysis_job)
    monkeypatch.setattr(api, "run_analysis_job", FakeTask())

    csv_bytes = (
        b"Q4_1,Q7_4,Q2_9,x1\n"
        b"1,1,10,0.1\n"
        b"2,2,20,0.2\n"
        b"3,3,30,0.3\n"
    )

    response = client.post(
        "/api/estimate",
        files={"file": ("valid.csv", csv_bytes, "text/csv")},
        data={"config": json.dumps(config)},
    )

    assert response.status_code == 202
    assert response.json() == {
        "job_id": str(expected_job_id),
        "status": "queued",
    }

    assert captured["filename"] == "valid.csv"
    assert captured["queued_job_id"] == str(expected_job_id)
    assert captured["input_path"] == str(captured["saved_path"])
    assert Path(captured["saved_path"]).read_bytes() == csv_bytes

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }
    
def test_estimate_rejects_overlapping_column_roles(
    config
):
    config["covariates"]["columns"].append(
        "Q4_1"
    )

    response = client.post(
        "/api/estimate",
        files={
            "file": (
                "data.csv",
                b"x\n1\n",
                "text/csv"
            )
        },
        data={
            "config": json.dumps(config)
        }
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        "covariatesには" in error["msg"]
        for error in errors
    )


def test_estimate_rejects_malformed_config_json():
    response = client.post(
        "/api/estimate",
        files={
            "file": (
                "data.csv",
                b"x\n1\n",
                "text/csv"
            )
        },
        data={
            "config": "{invalid-json"
        }
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["type"] == "json_invalid"
        for error in errors
    )


def test_estimate_requires_config():
    response = client.post(
        "/api/estimate",
        files={
            "file": (
                "data.csv",
                b"x\n1\n",
                "text/csv"
            )
        }
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"][-1] == "config"
        for error in errors
    )
    
def test_estimate_rejects_invalid_missing_config(
    config,
):
    config["missing"] = {
        "strategy": "fill",
    }

    response = client.post(
        "/api/estimate",
        files={
            "file": (
                "data.csv",
                b"x\n1\n",
                "text/csv",
            ),
        },
        data={
            "config": json.dumps(config),
        },
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        "fill_valuesを指定" in error["msg"]
        for error in errors
    )
    
def test_get_analysis_job_returns_404():
    response = client.get(f"/api/jobs/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "分析ジョブが見つかりません。"
    
def test_get_analysis_job_returns_saved_result():
    expected_result = {
        "segment": {"cut1": 1.0, "cut2": 2.0},
        "ate": [],
        "drcdf": [],
        "hei": None,
        "split_random_state": 123,
    }

    with SessionLocal() as db:
        job = create_analysis_job(
            db=db,
            config={},
            filename="sample.csv",
        )
        job_id = job.id
        mark_analysis_succeeded(db, job_id, expected_result)

    response = client.get(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == str(job_id)
    assert body["status"] == "succeeded"
    assert body["input_filename"] == "sample.csv"
    assert body["result"] == expected_result
    assert body["error_message"] is None