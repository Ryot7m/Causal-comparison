import logging
from pathlib import Path
from uuid import UUID
from datetime import datetime, timezone, timedelta

import pandas as pd
from app.analysis import run_analysis
from app.celery_app import celery_app
from app.database import SessionLocal
from app.dantic import AnalysisRequest
from app.models import AnalysisJob
from app.renew import mark_analysis_failed, mark_analysis_succeeded, recover_stale_analysis_jobs as recover_stale_jobs_in_db
from app.storage import csv_delete

logger = logging.getLogger(__name__)


@celery_app.task(
    name="app.tasks.run_analysis_job",
    ignore_result=True,
)
def run_analysis_job(job_id: str) -> None:
    job_uuid = UUID(job_id)
    db = SessionLocal()
    input_path = None

    try:
        job = db.get(AnalysisJob, job_uuid)
        if job is None:
            raise ValueError(f"分析ジョブが見つかりません: {job_id}")

        input_path = job.input_path  # ここで控える

        if not input_path:
            raise ValueError("ジョブにCSVの保存先がありません。")

        csv_path = Path(input_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.error_message = None
        db.commit()

        request_config = AnalysisRequest.model_validate(job.config)
        data = pd.read_csv(csv_path)
        result = run_analysis(data, request_config)

        mark_analysis_succeeded(db, job_uuid, result)

    except Exception as exc:
        db.rollback()

        if db.get(AnalysisJob, job_uuid) is not None:
            mark_analysis_failed(db, job_uuid, str(exc))

        logger.exception("Analysis job failed: %s", job_id)
        raise

    finally:
        if input_path:
            try:
                csv_delete(input_path)

                saved_job = db.get(AnalysisJob, job_uuid)
                if saved_job is not None:
                    saved_job.input_path = None
                    db.commit()

            except Exception:
                db.rollback()
                logger.exception(
                    "アップロードCSVの削除に失敗しました。job_id=%s",
                    job_id,
                )

        db.close()
        
    @celery_app.task(
    name="app.tasks.recover_stale_analysis_jobs",
    ignore_result=True,
    )
    def recover_stale_analysis_jobs_task(
        stale_after_minutes: int = 60,
    ) -> None:
        stale_before = datetime.now(timezone.utc) - timedelta(
            minutes=stale_after_minutes
        )

        with SessionLocal() as db:
            recovered_count = recover_stale_jobs_in_db(
                db,
                stale_before=stale_before,
            )

        logger.info("回復したstaleジョブ数: %s", recovered_count)