from dantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_column: str
    feature_columns: list[str] = Field(min_length=1)
    categorical_columns: list[str] = Field(default_factory=list)
    target_scores: dict[str, float] | None = None
    random_state: int = Field(default=123, ge=0, le=2**32 - 1)