import numpy as np
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session
from app.models import AnalysisJob

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

def create_analysis_job(db: Session, config: dict, filename: str) -> AnalysisJob:
    job = AnalysisJob(
        status="running",
        config=config,
        input_filename=filename,
    )
    
    db.add(job)
    db.commit()
    db.refresh(job)
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