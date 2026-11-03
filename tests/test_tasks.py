import pytest

from app.database import SessionLocal
from app.models import AnalysisJob
from app.renew import create_analysis_job
from app.tasks import run_analysis_job


def test_job_fails_when_required_columns_are_missing(config, tmp_path):
    csv_path = tmp_path / "missing_columns.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with SessionLocal() as db:
        job = create_analysis_job(
            db=db,
            config=config,
            filename="missing_columns.csv",
            input_path=str(csv_path)
        )
        job_id = job.id

    # タスクを直接実行するため、RedisやCeleryワーカーは不要
    with pytest.raises(ValueError, match="必要な列"):
        run_analysis_job.run(str(job_id))

    with SessionLocal() as db:
        job = db.get(AnalysisJob, job_id)

        assert job is not None
        assert job.status == "failed"
        assert "必要な列" in job.error_message
        assert job.finished_at is not None