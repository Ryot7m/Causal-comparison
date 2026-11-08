import numpy as np
from datetime import datetime, timezone
from uuid import UUID
import logging

from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session
from app.models import AnalysisJob
from app.storage import csv_delete

logger = logging.getLogger(__name__)

def to_json_compatible(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_json_compatible(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    return value

def create_analysis_job(db: Session, config: dict, filename: str,  input_path: str | None = None) -> AnalysisJob:
    job = AnalysisJob(
        status="queued",
        config=config,
        input_filename=filename,
        input_path=input_path
    )
    
    db.add(job)
    try:
        db.commit()
        db.refresh(job)
    except Exception:
        db.rollback()
        raise
    return job


def mark_analysis_succeeded(db: Session, job_id: UUID, result: dict) -> None:
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise ValueError(f"分析ジョブが見つかりません: {job_id}")

    job.status = "succeeded"
    job.result = to_json_compatible(result)
    job.finished_at = datetime.now(timezone.utc)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def mark_analysis_failed(db: Session, job_id: UUID, error_message: str) -> None:
    db.rollback()
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise ValueError(f"分析ジョブが見つかりません: {job_id}")
    job.status = "failed"
    job.error_message = error_message
    job.finished_at = datetime.now(timezone.utc)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    
def get_analysis_job(db: Session, job_id: UUID) -> AnalysisJob | None:
    return db.get(AnalysisJob, job_id)

def recover_stale_analysis_jobs(db: Session, stale_before: datetime) -> int:
    """期限を過ぎたrunning/queuedジョブを失敗扱いにする。"""
    now = datetime.now(timezone.utc)

    statement = (
        select(AnalysisJob)
        .where(
            or_(
                and_(
                    AnalysisJob.status == "running",
                    AnalysisJob.started_at.is_not(None),
                    AnalysisJob.started_at < stale_before,
                ),
                and_(
                    AnalysisJob.status == "queued",
                    AnalysisJob.created_at < stale_before,
                ),
            )
        )
        .with_for_update(skip_locked=True)
    )
    jobs = db.scalars(statement).all()

    files_to_delete = []
    for job in jobs:
        if job.status == "queued":
            job.error_message = (
                "分析ジョブが規定時間内にworkerへ取り込まれませんでした。"
            )
        else:
            job.error_message = (
                "分析workerが規定時間内に完了しませんでした。"
            )

        job.status = "failed"
        job.finished_at = now

        if job.input_path:
            files_to_delete.append((job.id, job.input_path))

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    # 失敗状態をDBに保存した後でCSVを削除する
    for job_id, path in files_to_delete:
        try:
            csv_delete(path)

            job = db.get(AnalysisJob, job_id)
            if job is not None and job.status == "failed":
                job.input_path = None
                db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "staleジョブのCSV削除に失敗しました。job_id=%s",
                job_id,
            )

    return len(jobs)