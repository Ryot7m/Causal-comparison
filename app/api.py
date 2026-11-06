from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Annotated
from app.dantic import AnalysisRequest, AnalysisJobAcceptedResponse
import logging
from fastapi import Depends
from sqlalchemy.orm import Session
from app.renew import create_analysis_job, mark_analysis_failed
from app.database import get_db
from uuid import UUID
from app.dantic import AnalysisJobResponse
from app.renew import get_analysis_job
from app.storage import csv_save
from app.tasks import run_analysis_job

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Estimate"]
)

@router.post("/estimate",  response_model=AnalysisJobAcceptedResponse, status_code=202)

async def estimate(file: Annotated[UploadFile, File(...)], config: Annotated[str, Form(...)], db: Annotated[Session, Depends(get_db)]):  
    try:
        request_config = (
            AnalysisRequest.model_validate_json(config)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=e.errors(include_context=False)
        ) from e
    
    contents = await file.read()
    input_path = csv_save(contents)
     
    job = create_analysis_job(
    db=db,
    config=request_config.model_dump(mode="json"),
    filename=file.filename or "uploaded.csv",
    input_path=input_path
    )
    
    try:
        run_analysis_job.delay(str(job.id))
    except Exception as exc:
        logger.exception("Failed to enqueue analysis job")
        mark_analysis_failed(
            db,
            job.id,
            "分析ジョブをキューに登録できませんでした。",
        )
        raise HTTPException(
            status_code=503,
            detail="分析ジョブをキューに登録できませんでした。",
        ) from exc

    return {
        "job_id": job.id,
        "status": "queued",
    }
        
@router.get("/jobs/{job_id}", response_model=AnalysisJobResponse)

def read_analysis_job(
    job_id: UUID,
    db: Annotated[Session, Depends(get_db)],
):
    job = get_analysis_job(db, job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="分析ジョブが見つかりません。",
        )

    return {
        "job_id": job.id,
        "status": job.status,
        "input_filename": job.input_filename,
        "result": job.result,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }