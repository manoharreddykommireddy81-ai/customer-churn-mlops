
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    model = load_model()

    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    elif isinstance(customer_data, list):
        customer_data = pd.DataFrame(customer_data)

    prediction = int(model.predict(customer_data)[0])
    probability = float(model.predict_proba(customer_data)[0][1])

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "result": "Churn" if prediction == 1 else "No Churn"
    } 
