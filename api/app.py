from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API",version="1.0.0")

model = joblib.load("models/churn_model.pkl")

class CustomerRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
@app.get("/")
def health():
    return {"status": "running","model": "customer churn prediction"}

@app.post("/predict")
def predict(customer: CustomerRequest):
    input_df = pd.DataFrame([customer.dict()])
    prediction = model.predict(input_df)[0]
    probability = (model.predict_proba(input_df)[0][1])

    return {
        "prediction": int(prediction),
        "churn_probability": round(
            float(probability),
            4
        ),
        "label": (
            "Likely To Churn"
            if prediction == 1
            else "Likely To Stay"
        )
    }