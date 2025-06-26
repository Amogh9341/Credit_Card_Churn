from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap

app = FastAPI()

# === Load Model and Scaler ===
model = joblib.load("../model/churn_model.pkl")  # Ensure file is in same folder
scaler = joblib.load("../model/scaler.pkl")

# === Columns to Scale ===
SCALE_FEATURES = [
    'Customer_Age', 'Months_on_book', 'Total_Relationship_Count',
    'Months_Inactive_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Contacts_Count_12_mon', 'Avg_Trans_Amt', 'Monthly_Rev_Bal'
]

# === Input Data Format ===
class ChurnInput(BaseModel):
    Customer_Age: int
    Gender: int
    Dependent_count: int
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: float
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: float
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float
    Marital_Status_Divorced: int
    Marital_Status_Married: int
    Marital_Status_Single: int
    Avg_Trans_Amt: float
    Monthly_Rev_Bal: float
    Engagement_Drop: float
    Spending_Drop: float

# === Preprocessing Function ===
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    df_scaled = df.copy()
    df_scaled[SCALE_FEATURES] = scaler.transform(df[SCALE_FEATURES])
    return df_scaled

# === Predict Route ===
@app.post("/predict")
def predict_churn(data: ChurnInput):
    processed_df = preprocess_input(data.model_dump())
    input_array = processed_df.values
    prob = model.predict_proba(input_array)[0][1]
    return {"churn_probability": float(prob)}

# === SHAP Explainer ===
explainer = shap.TreeExplainer(model)

# === SHAP Explain Route ===
@app.post("/explain")
def explain_churn(data: ChurnInput):
    input_df = preprocess_input(data.model_dump())
    shap_values = explainer(input_df)

    # Handle binary classifier SHAP output
    shap_contributions = shap_values.values[0]
    if shap_contributions.ndim > 1:
        shap_contributions = shap_contributions[:, 1]

    base_value = shap_values.base_values[0]
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])
    else:
        base_value = float(base_value)

    top_contributors = sorted(
        zip(input_df.columns.tolist(), shap_contributions),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return {
        "base_value": base_value,
        "shap_values": shap_contributions.tolist(),
        "feature_names": input_df.columns.tolist(),
        "feature_values": input_df.iloc[0].tolist(),
        "top_contributors": top_contributors[:5]
    }

