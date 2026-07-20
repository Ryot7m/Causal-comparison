from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import EstimateResponse
from app.services import estimate_service

router = APIRouter(
    prefix="/api",
    tags=["Estimate"]
)

@router.post("/estimate", response_model=EstimateResponse)

async def estimate(file : UploadFile = File(...)):
    
    try:
        result = await estimate_service(file)
        return result

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