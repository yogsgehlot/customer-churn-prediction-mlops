import pandas as pd
import joblib


model = joblib.load("models/churn_model.pkl")

sample_customer = pd.DataFrame([
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 450.5
    }
])

prediction = model.predict(sample_customer)

probability = model.predict_proba(sample_customer)

print("Prediction:",prediction[0])

print("Probability:",probability[0][1])