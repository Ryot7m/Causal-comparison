from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import EstimateResponse
from app.services import estimate_service

app = APIRouter()

@router.post(
    "/estimate",
    response_model=EstimateResponse
)
async def estimate(file : UploadFile = File("sample.csv")):
    
    try:
        result = await estimate_service(file)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )