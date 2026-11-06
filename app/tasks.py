import logging
from pathlib import Path
from uuid import UUID
import pandas as pd
import pytest

from app.analysis import run_analysis
from app.celery_app import celery_app
from app.database import SessionLocal
from app.dantic import AnalysisRequest
from app.models import AnalysisJob
from app.renew import mark_analysis_failed, mark_analysis_succeeded, create_analysis_job
import app.storage as astg

logger = logging.getLogger(__name__)


@celery_app.task(
    name="app.tasks.run_analysis_job",
    ignore_result=True,
)
def run_analysis_job(job_id: str) -> None:
    job_uuid = UUID(job_id)
    db = SessionLocal()

    try:
        job = db.get(AnalysisJob, job_uuid)
        if job is None:
            raise ValueError(f"分析ジョブが見つかりません: {job_id}")

        if not job.input_path:
            raise ValueError("ジョブにCSVの保存先がありません。")

        csv_path = Path(job.input_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

        job.status = "running"
        job.error_message = None
        db.commit()

        request_config = AnalysisRequest.model_validate(job.config)
        data = pd.read_csv(csv_path)
        result = run_analysis(data, request_config)

        mark_analysis_succeeded(db, job_uuid, result)

    except Exception as exc:
        db.rollback()

        # ジョブが存在する場合は、失敗状態をDBに保存する
        if db.get(AnalysisJob, job_uuid) is not None:
            mark_analysis_failed(db, job_uuid, str(exc))

        logger.exception("Analysis job failed: %s", job_id)
        raise

    finally:
        db.close()
        
def test_job_fails_when_required_columns_are_missing(
    config,
    tmp_path,
    monkeypatch,
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(astg, "UPLOAD_DIR", upload_dir)

    csv_path = upload_dir / "missing_columns.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with SessionLocal() as db:
        job = create_analysis_job(
            db=db,
            config=config,
            filename="missing_columns.csv",
            input_path=str(csv_path),
        )
        job_id = job.id

    with pytest.raises(ValueError, match="必要な列"):
        run_analysis_job.run(str(job_id))

    assert not csv_path.exists()

    with SessionLocal() as db:
        job = db.get(AnalysisJob, job_id)
        assert job is not None
        assert job.status == "failed"
        assert "必要な列" in job.error_message
        assert job.finished_at is not None
        assert job.input_path is None