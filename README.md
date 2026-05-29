# 🚀 AI-Powered Customer Churn Prediction System

An end-to-end Machine Learning and MLOps project that predicts customer churn using the IBM Telco Customer Churn dataset.

This project follows a complete production-grade machine learning workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, SMOTE-based class imbalance handling, model training, MLflow experiment tracking, FastAPI deployment, and Docker containerization.

---

## 📌 Business Problem

Customer churn directly impacts business revenue and growth.

Acquiring a new customer is often more expensive than retaining an existing one. The goal of this project is to identify customers who are likely to leave so businesses can proactively implement retention strategies.

---

## 🎯 Project Objectives

* Predict customer churn using machine learning.
* Identify key factors driving customer churn.
* Compare multiple machine learning models.
* Improve churn detection using SMOTE.
* Expose predictions through a FastAPI REST API.
* Track experiments using MLflow.
* Containerize the application using Docker.

---

## 🏗️ Project Architecture

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Engineering
   │
   ▼
Data Preprocessing
   │
   ▼
SMOTE
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
MLflow Tracking
   │
   ▼
Model Serialization
   │
   ▼
FastAPI Service
   │
   ▼
Docker Deployment
```

---

## 📂 Project Structure

```text
customer-churn-prediction-mlops/

│
├── api/
│   └── app.py
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│
├── mlruns/
│
├── requirements.txt
│
├── Dockerfile
│
└── README.md
```

---

## 📊 Dataset Information

Dataset: IBM Telco Customer Churn Dataset

Total Records:

```text
7043 Customers
```

Features:

```text
21 Features
```

Target Variable:

```text
Churn

0 → Customer Stays
1 → Customer Leaves
```

Dataset Source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 🔍 Exploratory Data Analysis Findings

### High-Risk Customers

* Month-to-Month Contract Customers
* Fiber Optic Internet Users
* Electronic Check Payment Users
* Customers Without Tech Support
* Customers Without Online Security

### Low-Risk Customers

* Long Tenure Customers
* Two-Year Contract Customers
* Customers With Dependents
* DSL Customers

---

## 🤖 Machine Learning Models

### Random Forest

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 78.75% |
| Precision | 63.07% |
| Recall    | 48.40% |
| F1 Score  | 54.77% |
| ROC-AUC   | 0.814  |

---

### Logistic Regression

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 80.38% |
| Precision | 64.85% |
| Recall    | 57.22% |
| F1 Score  | 60.80% |
| ROC-AUC   | 0.836  |

---

### SMOTE + Logistic Regression (Selected Model)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 73.13% |
| Precision | 49.66% |
| Recall    | 77.81% |
| F1 Score  | 60.63% |
| ROC-AUC   | 0.833  |

---

## 🏆 Why This Model Was Selected

Although Logistic Regression achieved higher accuracy, customer churn prediction prioritizes Recall over Accuracy.

Missing a churn customer is significantly more costly than contacting a customer who would have stayed.

The SMOTE-enhanced Logistic Regression model successfully identified approximately:

```text
77.81% of customers likely to churn
```

making it the most business-effective solution.

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* Imbalanced-Learn (SMOTE)

### API Development

* FastAPI

### MLOps

* MLflow

### Deployment

* Docker

---

## 🚀 Installation

Clone Repository

```bash
git clone https://github.com/yogsgehlot/customer-churn-prediction-mlops.git

cd customer-churn-prediction-mlops
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train Model

```bash
python src/train.py
```

Output:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

Model saved to:

```text
models/churn_model.pkl
```

---

## 🔮 Run Prediction Script

```bash
python src/predict.py
```

Example Output

```text
Prediction: 1
Probability: 0.87
```

---

## 🌐 Run FastAPI Application

```bash
uvicorn api.app:app --reload
```

Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

---

## 📮 API Example

### POST /predict

Request

```json
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
```

Response

```json
{
  "prediction": 1,
  "churn_probability": 0.87,
  "label": "Likely To Churn"
}
```

---

## 📈 MLflow Experiment Tracking

Start MLflow UI

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Tracked Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Tracked Artifacts:

* Trained Models
* Parameters
* Experiment Runs

---

## 🐳 Docker

Build Docker Image

```bash
docker build -t churn-api .
```

Run Container

```bash
docker run -p 8000:8000 churn-api
```

---

## 💼 Resume Highlights

* Built an end-to-end Customer Churn Prediction System using Machine Learning and MLOps practices.
* Performed advanced EDA and identified key churn-driving customer behaviors.
* Applied SMOTE to address class imbalance and improve churn recall from 57% to 78%.
* Developed and compared multiple machine learning models using Scikit-Learn.
* Implemented MLflow-based experiment tracking and model versioning.
* Created production-ready FastAPI inference APIs and Dockerized deployment architecture.

---

## 🔮 Future Improvements

* XGBoost Integration
* LightGBM Integration
* Model Monitoring
* CI/CD Pipeline
* AWS Deployment
* Kubernetes Deployment
* Real-Time Prediction Dashboard

---

## 👨‍💻 Author

Yogesh Gehlot

If you found this project useful, consider giving it a ⭐ on GitHub.


