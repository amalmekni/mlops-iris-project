from fastapi.testclient import TestClient
from app.main import app, _target_names

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_single():
    sample = {"features": [5.1, 3.5, 1.4, 0.2]}
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    data = r.json()
    assert "labels" in data and len(data["labels"]) == 1
    assert data["classes"] == _target_names
