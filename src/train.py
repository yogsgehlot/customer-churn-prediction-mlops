import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
# Load Data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data Cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)
df["Churn"] = df["Churn"].map({"No": 0,"Yes": 1})

# Features & Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Feature Types
categorical_features = X.select_dtypes(include=["object"]).columns
numeric_features = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Model Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

mlflow.set_experiment("Customer Churn Prediction")

with mlflow.start_run():
            
    # Train
    pipeline.fit(X_train, y_train)

    # Prediction
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    auc = roc_auc_score(y_test,y_prob)
    
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1",f1)
    mlflow.log_metric("auc",auc)
    mlflow.sklearn.log_model(pipeline,"churn_model")
    
# Metrics
print("=" * 50)
print("Accuracy:",accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred))
print("Recall:",recall_score(y_test, y_pred))
print("F1:",f1_score(y_test, y_pred))
print("ROC AUC:",roc_auc_score(y_test, y_prob))
print("=" * 50)

# Save Model
joblib.dump(pipeline,"models/churn_model.pkl")
print("Model Saved Successfully")