from datetime import datetime
from math import isfinite
import pandas as pd
from typing import Literal
from uuid import UUID

from app.dantic import Scalar, BaseModel, ConfigDict, Field, model_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_column: str
    feature_columns: list[str] = Field(min_length=1)
    categorical_columns: list[str] = Field(default_factory=list)

    # 数値の目的変数なら両方None。
    # 順序カテゴリなら、levelsとscoresを両方指定する。
    target_levels: list[Scalar] | None = None
    target_scores: list[float] | None = None

    random_state: int = Field(default=123, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def validate_request(self):
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("feature_columnsに重複があります。")

        if len(self.categorical_columns) != len(set(self.categorical_columns)):
            raise ValueError("categorical_columnsに重複があります。")

        unknown = set(self.categorical_columns) - set(self.feature_columns)
        if unknown:
            raise ValueError(
                f"categorical_columnsに特徴量以外の列があります: {sorted(unknown)}"
            )

        if (self.target_levels is None) != (self.target_scores is None):
            raise ValueError(
                "target_levelsとtarget_scoresは両方指定するか、両方省略してください。"
            )

        if self.target_levels is not None:
            if len(self.target_levels) != len(self.target_scores):
                raise ValueError(
                    "target_levelsとtarget_scoresの長さを一致させてください。"
                )

            if len(set(self.target_levels)) != len(self.target_levels):
                raise ValueError("target_levelsに重複があります。")

            if not all(isfinite(score) for score in self.target_scores):
                raise ValueError("target_scoresには有限値を指定してください。")

            if any(
                left >= right
                for left, right in zip(
                    self.target_scores,
                    self.target_scores[1:],
                )
            ):
                raise ValueError("target_scoresは昇順にしてください。")

        return self
        
class PredictionInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Scalar | None]] = Field(
        min_length=1,
        max_length=1000,
    )

class PredictionInferenceResponse(BaseModel):
    model_id: UUID
    predictions: list[float]
    
class PredictionJobAcceptedResponse(BaseModel):
    job_id: UUID
    status: Literal["queued"]

class PredictionJobResponse(BaseModel):
    job_id: UUID
    status: Literal["queued", "running", "succeeded", "failed"]
    input_filename: str | None = None
    result: dict | None = None
    error_message: str | None = None
    created_at: datetime
    finished_at: datetime | None = None