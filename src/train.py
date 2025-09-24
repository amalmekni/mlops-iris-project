from pathlib import Path
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "metadata.json"

def train_and_save(model_dir: Path = MODEL_DIR):
    data = load_iris(as_frame=False)
    X, y = data.data, data.target
    target_names = list(data.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])
    pipe.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipe.predict(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    META_PATH.write_text(json.dumps({"target_names": target_names}, indent=2))
    return acc, pipe, target_names

if __name__ == "__main__":
    acc, _, _ = train_and_save()
    print(f"✅ Model trained with accuracy={acc:.3f}, saved to {MODEL_PATH}")
