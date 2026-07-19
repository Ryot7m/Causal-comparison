from fastapi import FastAPI

from app.config import (
    API_TITLE,
    API_VERSION
)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
)

@app.get("/health")
def health():
    return {"status": "ok"}