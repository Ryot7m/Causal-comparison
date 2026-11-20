import os
from pathlib import Path
from uuid import UUID

MODEL_DIR = Path(os.getenv("MODEL_DIR", "data/models"))


def model_path_for(model_id: UUID) -> Path:
    return MODEL_DIR / f"{model_id}.joblib"