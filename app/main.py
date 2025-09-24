from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import joblib, json
import numpy as np

app = FastAPI(title="Iris FastAPI", version="1.0.0")

MODEL_PATH = Path("model/model.joblib")
META_PATH = Path("model/metadata.json")

# Charger modèle et classes
if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run: python src/train.py")

_model = joblib.load(MODEL_PATH)
_target_names = json.loads(META_PATH.read_text())["target_names"]

class PredictIn(BaseModel):
    features: Optional[List[float]] = Field(
        default=None,
        description="[sepal_len, sepal_wid, petal_len, petal_wid]"
    )
    instances: Optional[List[List[float]]] = Field(
        default=None,
        description="Batch of samples"
    )

class PredictOut(BaseModel):
    labels: List[str]
    probabilities: List[List[float]]
    classes: List[str]

@app.get("/")
def root():
    return {"message": "Iris API is running 🚀. Use /docs to test, /health to check status."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    if payload.instances is None and payload.features is None:
        raise HTTPException(status_code=400, detail="Provide 'features' or 'instances'")

    # ✅ ICI le bloc NumPy (remplace l'ancien bloc pandas)
    if payload.instances is None:
        X = np.array([payload.features], dtype=float)
    else:
        X = np.array(payload.instances, dtype=float)

    probs = _model.predict_proba(X).tolist()
    labels = [_target_names[int(np.argmax(p))] for p in probs]
    return PredictOut(labels=labels, probabilities=probs, classes=_target_names)
