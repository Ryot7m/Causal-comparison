import os
from pathlib import Path
from uuid import uuid4


UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))


def csv_save(contents: bytes) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    path = UPLOAD_DIR / f"{uuid4().hex}.csv"
    path.write_bytes(contents)

    return str(path)