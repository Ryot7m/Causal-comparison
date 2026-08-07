import pandas as pd
from app import analysis

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

def test_estimate_accepts_valid_csv(monkeypatch):
    captured = {}

    def fake_estimate(data, config):
        captured["columns"] = list(data.columns)
        captured["row_count"] = len(data)

        return {
            "segment": {
                "cut1": 1.0,
                "cut2": 2.0,
            },
            "ate": {
                "res": pd.DataFrame([
                    {
                        "cls": 0,
                        "clsnum": 3,
                        "ate": 0.2,
                        "se": 0.05,
                        "ci_low": 0.1,
                        "ci_high": 0.3,
                    },
                ]),
            },
            "drcdf": pd.DataFrame([
                {
                    "seg": 0,
                    "c": 0,
                    "threshold": 1.0,
                    "F1_dr": 0.4,
                    "F0_dr": 0.5,
                    "tau_c": -0.1,
                    "se_c": 0.05,
                    "ci_low": -0.2,
                    "ci_high": 0.0,
                },
            ]),
            "hei": 0.25,
        }

    monkeypatch.setattr(
        analysis,
        "estimate",
        fake_estimate,
    )

    csv = (
        "Q4_1,Q7_4,Q2_9,x1\n"
        "1,1,10,0.1\n"
        "2,2,20,0.2\n"
        "3,3,30,0.3\n"
    ).encode("utf-8")

    response = client.post(
        "/api/estimate",
        files={
            "file": (
                "valid.csv",
                csv,
                "text/csv",
            ),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert captured["row_count"] == 3
    assert captured["columns"] == [
        "Q4_1",
        "Q7_4",
        "Q2_9",
        "x1",
    ]

    assert body["segment"] == {
        "cut1": 1.0,
        "cut2": 2.0,
    }

    assert len(body["ate"]) == 1
    assert len(body["drcdf"]) == 1
    assert body["hei"] == {
        "score": 0.25,
    }

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_estimate_rejects_missing_columns():
    csv = b"a,b\n1,2\n"

    response = client.post(
        "/api/estimate",
        files={
            "file": (
                "invalid.csv",
                csv,
                "text/csv",
            ),
        },
    )

    assert response.status_code == 400
    assert "必要な列" in response.json()["detail"]