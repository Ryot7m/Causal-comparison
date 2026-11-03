import logging
from pathlib import Path
from uuid import UUID

import pandas as pd

from app.analysis import run_analysis
from app.celery_app import celery_app
from app.database import SessionLocal
from app.dantic import AnalysisRequest
from app.models import AnalysisJob
from app.renew import mark_analysis_failed, mark_analysis_succeeded

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