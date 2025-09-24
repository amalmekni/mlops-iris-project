from src.train import train_and_save, MODEL_PATH

def test_training_and_persist():
    acc, model, _ = train_and_save()
    assert acc > 0.9
    assert MODEL_PATH.exists()
    assert hasattr(model, "predict_proba")
