# app/prediction_tasks.py の処理イメージ
# app/prediction_tasks.py
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pandas as pd

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models import PredictionJob
from app.predict_dantic import PredictionRequest
from app.predict_storage import MODEL_DIR, model_path_for
from app.renew import to_json_compatible
from app.storage import csv_delete
from predict.prediction import train_prediction_model

logger = logging.getLogger(__name__)


def _remove_partial_model(path: Path | None) -> None:
    """失敗時に作られたモデルだけをMODEL_DIR内で削除する。"""
    if path is None:
        return

    try:
        model_root = MODEL_DIR.resolve()
        target = path.resolve()

        if not target.is_relative_to(model_root):
            raise ValueError("MODEL_DIR外のモデルは削除できません。")

        target.unlink(missing_ok=True)
    except Exception:
        logger.exception("部分モデルの削除に失敗しました: %s", path)


@celery_app.task(
    name="app.prediction_tasks.run_prediction_training_job",
    ignore_result=True,
)
def run_prediction_training_job(job_id: str) -> None:
    job_uuid = UUID(job_id)
    db = SessionLocal()
    input_path: str | None = None
    model_path: Path | None = None

    try:
        job = db.get(PredictionJob, job_uuid)
        if job is None:
            raise ValueError(f"予測ジョブが見つかりません: {job_id}")

        # finallyで削除するため、先に保存パスを控える
        input_path = job.input_path
        if not input_path:
            raise ValueError("ジョブに入力CSVの保存先がありません。")

        csv_path = Path(input_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"入力CSVが見つかりません: {csv_path}")

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.error_message = None
        db.commit()

        config = PredictionRequest.model_validate(job.config)
        data = pd.read_csv(csv_path)

        # UUIDから決めるパスなので、ユーザー入力のファイル名は使わない
        model_path = model_path_for(job_uuid)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        result = train_prediction_model(
            data,
            target_column=config.target_column,
            feature_columns=config.feature_columns,
            categorical_columns=config.categorical_columns,
            target_scores=config.target_scores,
            random_state=config.random_state,
            model_path=str(model_path),
        )

        # stale回復などで既に失敗扱いになったジョブを成功に戻さない
        job = db.get(PredictionJob, job_uuid)
        if job is None or job.status != "running":
            raise RuntimeError("予測ジョブが実行中ではありません。")

        job.model_path = str(model_path)
        job.result = to_json_compatible({
            "model_id": str(job_uuid),
            "metrics": result["metrics"],
            "feature_importance": result["feature_importance"],
        })
        job.status = "succeeded"
        job.finished_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        db.rollback()
        _remove_partial_model(model_path)

        try:
            job = db.get(PredictionJob, job_uuid)
            if job is not None and job.status != "succeeded":
                job.status = "failed"
                job.error_message = str(exc)
                job.finished_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "予測ジョブの失敗状態を保存できませんでした: %s",
                job_id,
            )

        logger.exception("Prediction job failed: %s", job_id)
        raise

    finally:
        if input_path:
            try:
                csv_delete(input_path)

                job = db.get(PredictionJob, job_uuid)
                if job is not None:
                    job.input_path = None
                    db.commit()
            except Exception:
                db.rollback()
                logger.exception(
                    "予測ジョブの入力CSVを削除できませんでした: %s",
                    job_id,
                )

        db.close()