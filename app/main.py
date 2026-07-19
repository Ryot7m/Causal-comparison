from fastapi import FastAPI
from app.api import router

app = FastAPI(
    title="Causal Inference Platform",
    version="1.0.0"
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}