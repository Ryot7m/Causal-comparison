import os
import sys

print("cwd =", os.getcwd())
print("sys.path =", sys.path)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import router

from app.config import (
    API_TITLE,
    API_VERSION
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("API started successfully!")
    print("Swagger UI: http://localhost:8080/docs")
    print("=" * 50)

    yield

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    docs_url="/docs",
    lifespan=lifespan
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}