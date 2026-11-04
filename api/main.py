
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.predict import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using a trained ML model",
    version="1.0.0"
)


class CustomerData(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        result = predict_churn(customer.model_dump())
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) 
