import os
import sys

print("cwd =", os.getcwd())
print("sys.path =", sys.path)

from fastapi import FastAPI
from app.api import router

from app.config import (
    API_TITLE,
    API_VERSION
)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}