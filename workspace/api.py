from fastapi import FastAPI
import pandas as pd

from src.aipw import estimate_aipw

app = FastAPI()

@app.post("/estimate")
def estimate():

    df = pd.read_csv("sample.csv")

    ate = estimate_aipw(df)

    return {
        "ATE": ate
    }