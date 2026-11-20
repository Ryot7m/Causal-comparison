import os
from pathlib import Path
from uuid import uuid4


UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))


def csv_save(contents: bytes) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    path = UPLOAD_DIR / f"{uuid4().hex}.csv"
    path.write_bytes(contents)
    return str(path)
    return str(path)

def csv_delete(path: str | None) -> None:
    if not path:
        return

    upload_root = UPLOAD_DIR.resolve()
    target = Path(path).resolve()

    # UPLOAD_DIRの外にあるファイルは削除しない
    if not target.is_relative_to(upload_root):
        raise ValueError("UPLOAD_DIR外のファイルは削除できません。")

    target.unlink(missing_ok=True)
