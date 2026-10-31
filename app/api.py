from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Annotated
from app.dantic import AnalysisRequest,EstimateResponse
from app.analysis import estimate_service
import logging
from fastapi import Depends
from sqlalchemy.orm import Session
from app.renew import create_analysis_job,mark_analysis_succeeded,mark_analysis_failed
from app.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Estimate"]
)

@router.post("/estimate", response_model=EstimateResponse)

async def estimate(file: Annotated[UploadFile, File(...)], config: Annotated[str, Form(...)], db: Annotated[Session, Depends(get_db)]):  
    try:
        request_config = (
            AnalysisRequest.model_validate_json(config)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=e.errors(include_context=False)
        )
        
    job = create_analysis_job(
    db=db,
    config=request_config.model_dump(mode="json"),
    filename=file.filename or "uploaded.csv"
    )
    
    
    try:
        response = await estimate_service(file, request_config)
        mark_analysis_succeeded(db, job.id, response)
        return response
    
    except ValueError as exc:
        mark_analysis_failed(db, job.id, str(exc))
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
    
    except Exception as exc:
        logger.exception("estimate endpoint failed")
        mark_analysis_failed(db, job.id, "Internal Server Error")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        ) from exc