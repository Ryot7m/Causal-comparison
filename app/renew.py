from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import AnalysisJob


def create_analysis_job(
    db: Session,
    config: dict,
    filename: str,
) -> AnalysisJob:
    job = AnalysisJob(
        status="running",
        config=config,
        input_filename=filename,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def mark_analysis_succeeded(
    db: Session,
    job_id: UUID,
    result: dict,
) -> None:
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise ValueError(f"分析ジョブが見つかりません: {job_id}")

    job.status = "succeeded"
    job.result = result
    job.finished_at = datetime.now(timezone.utc)
    db.commit()


def mark_analysis_failed(
    db: Session,
    job_id: UUID,
    error_message: str,
) -> None:
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise ValueError(f"分析ジョブが見つかりません: {job_id}")

    job.status = "failed"
    job.error_message = error_message
    job.finished_at = datetime.now(timezone.utc)
    db.commit()