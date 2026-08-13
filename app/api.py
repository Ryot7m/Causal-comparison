from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Annotated
from app.dantic import AnalysisRequest,EstimateResponse
from app.analysis import estimate_service

router = APIRouter(
    prefix="/api",
    tags=["Estimate"]
)

@router.post("/estimate", response_model=EstimateResponse)

async def estimate(file: Annotated[UploadFile, File(...)], config: Annotated[str, Form(...)]):
    
    try:
        request_config = (
            AnalysisRequest.model_validate_json(config)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=e.errors(include_context=False)
        )
        
    try:
        return await estimate_service(
            file=file,
            request_config=request_config
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
        
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )