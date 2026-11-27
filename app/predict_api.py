import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
import pandas as pd
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.database import get_db
from app.models import PredictionJob
from app.predict_dantic import PredictionJobAcceptedResponse, PredictionJobResponse, PredictionInferenceRequest, PredictionInferenceResponse, PredictionRequest
from app.predict_storage import model_path_for
from app.storage import csv_delete, csv_save
from predict.prediction import predict_with_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prediction", tags=["Prediction"])


@router.post("/train", response_model=PredictionJobAcceptedResponse, status_code=202)
async def train_prediction(
    file: Annotated[UploadFile, File(...)],
    config: Annotated[str, Form(...)],
    db: Annotated[Session, Depends(get_db)],
):
    try:
        request_config = PredictionRequest.model_validate_json(config)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_context=False),
        ) from exc

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="CSVファイルが空です。")

    input_path = csv_save(contents)
    job = PredictionJob(
        status="queued",
        config=request_config.model_dump(mode="json"),
        input_filename=file.filename or "uploaded.csv",
        input_path=input_path,
    )
    db.add(job)

    try:
        db.commit()
        db.refresh(job)
    except Exception as exc:
        db.rollback()
        try:
            csv_delete(input_path)
        except Exception:
            logger.exception("ジョブ登録失敗後のCSV削除に失敗しました。")
        raise HTTPException(
            status_code=500,
            detail="予測ジョブを登録できませんでした。",
        ) from exc

    try:
        celery_app.send_task(
            "app.prediction_tasks.run_prediction_training_job",
            args=[str(job.id)],
        )
    except Exception as exc:
        logger.exception("予測ジョブをCeleryへ登録できませんでした。")
        try:
            job = db.get(PredictionJob, job.id)
            if job is not None and job.status == "queued":
                job.status = "failed"
                job.error_message = "予測ジョブをキューに登録できませんでした。"
                job.finished_at = datetime.now(timezone.utc)
                job.input_path = None
                db.commit()
        except Exception:
            db.rollback()
            logger.exception("予測ジョブの失敗状態を保存できませんでした。")

        try:
            csv_delete(input_path)
        except Exception:
            logger.exception("キュー登録失敗後のCSV削除に失敗しました。")
        raise HTTPException(
            status_code=503,
            detail="予測ジョブをキューに登録できませんでした。",
        ) from exc

    return {"job_id": job.id, "status": "queued"}


@router.get("/jobs/{job_id}", response_model=PredictionJobResponse)
def read_prediction_job(
    job_id: UUID,
    db: Annotated[Session, Depends(get_db)],
):
    job = db.get(PredictionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="予測ジョブが見つかりません。")

    return {
        "job_id": job.id,
        "status": job.status,
        "input_filename": job.input_filename,
        "result": job.result,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }


@router.post(
    "/models/{model_id}/predict",
    response_model=PredictionInferenceResponse,
)
def predict_with_saved_model(
    model_id: UUID,
    request: PredictionInferenceRequest,
    db: Annotated[Session, Depends(get_db)],
):
    job = db.get(PredictionJob, model_id)
    if job is None:
        raise HTTPException(status_code=404, detail="予測モデルが見つかりません。")
    if job.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail="学習が成功したモデルだけ推論に使用できます。",
        )

    artifact = model_path_for(model_id)
    if (
        not job.model_path
        or Path(job.model_path).resolve() != artifact.resolve()
        or not artifact.is_file()
    ):
        raise HTTPException(
            status_code=410,
            detail="学習済みモデルのファイルが利用できません。",
        )

    config = PredictionRequest.model_validate(job.config)
    expected_columns = set(config.feature_columns)
    for index, record in enumerate(request.records):
        actual_columns = set(record)
        if actual_columns != expected_columns:
            missing = sorted(expected_columns - actual_columns)
            extra = sorted(actual_columns - expected_columns)
            raise HTTPException(
                status_code=422,
                detail={
                    "row": index,
                    "missing_columns": missing,
                    "extra_columns": extra,
                },
            )

    try:
        predictions = predict_with_model(
        pd.DataFrame(request.records),
        model_path=str(artifact),
        feature_columns=config.feature_columns
    )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("保存済みモデルでの推論に失敗しました: %s", model_id)
        raise HTTPException(
            status_code=500,
            detail="予測の実行に失敗しました。",
        ) from exc

    return {"model_id": model_id, "predictions": predictions}
