from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


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