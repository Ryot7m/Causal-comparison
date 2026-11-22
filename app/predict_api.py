import pandas as pd
from pathlib import Path
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import PredictionJob
from app.predict_dantic import PredictionRequest, PredictionInferenceRequest, PredictionInferenceResponse
from app.predict_storage import model_path_for
from predict.prediction import predict_with_model

@router.post(
    "/models/{model_id}/predict",
    response_model=PredictionInferenceResponse,
)
def predict_with_saved_model(
    model_id: UUID,
    request: PredictionInferenceRequest,
    db: Annotated[Session, Depends(get_db)]
):
    job = db.get(PredictionJob, model_id)
    if job is None:
        raise HTTPException(status_code=404, detail="予測モデルが見つかりません。")
    if job.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail="学習が成功したモデルだけ推論に使用できます。"
        )

    artifact = model_path_for(model_id)
    if (
        not job.model_path
        or Path(job.model_path).resolve() != artifact.resolve()
        or not artifact.is_file()
    ):
        raise HTTPException(
            status_code=410,
            detail="学習済みモデルのファイルが利用できません。"
        )

    config = PredictionRequest.model_validate(job.config)
    expected_columns = set(config.feature_columns)

    for index, record in enumerate(request.records):
        actual_columns = set(record)
        if actual_columns != expected_columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "row": index,
                    "missing_columns": sorted(expected_columns - actual_columns),
                    "extra_columns": sorted(actual_columns - expected_columns),
                }
            )

    predictions = predict_with_model(
        pd.DataFrame(request.records),
        model_path=str(artifact),
        feature_columns=config.feature_columns,
        categorical_columns=config.categorical_columns
    )
    return {"model_id": model_id, "predictions": predictions}