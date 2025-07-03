from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
import json
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# === Load Model and Prepocessor ===
preprocessor = joblib.load('../artifacts/preprocessor.pkl')
model = joblib.load('../artifacts/churn_model.pkl')
with open('../artifacts/threshold.json') as f:
    threshold = json.load(f)['optimal_threshold']
with open('../artifacts/not_selected_feature_but_need_for_preprocessor.json') as f:
    extras = json.load(f)
with open('../artifacts/all_features.json') as f:
    all_features = json.load(f)
# Load feature order for consistent column selection
with open('../artifacts/feature_order.json') as f:
    feature_order = json.load(f)

class ChurnInput(BaseModel):
    Customer_Age: int
    Gender: int
    Dependent_count: int
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str
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
    Trans_Amt_per_Trans_Ct: float
    Trans_Amt_per_Rev_Bal: float
    Trans_Amt_per_Open_To_Buy: float
    Trans_Amt_per_Contacts: float
    Trans_Amt_per_Inactive: float
    Trans_Amt_per_Relationship: float
    Trans_Amt_per_Dependent: float
    Rev_Bal_per_Contacts: float
    Rev_Bal_per_Relationship: float
    Credit_Limit_per_Trans_Ct: float
    Credit_Limit_per_Inactive: float
    Credit_Limit_per_Relationship: float
    Credit_Limit_per_Dependent: float
    Amt_Chng_per_Trans_Ct: float
    Amt_Chng_per_Contacts: float
    Amt_Chng_per_Inactive: float
    Amt_Chng_per_Relationship: float
    Amt_Chng_per_Dependent: float
    Trans_Ct_per_Dependent: float
    Trans_Ct_per_Months_Inactive: float
    Trans_Ct_per_Relationship: float
    Sum_Trans_Amt_Rev_Bal: float
    Sum_Trans_Amt_Open_To_Buy: float

def preprocess_input(data):
    input_data = data.dict()
    for feature in extras:
        if feature not in input_data:
            input_data[feature] = 0
    X_full = preprocessor.transform(pd.DataFrame([input_data]))
    X = pd.DataFrame(X_full, columns=list(all_features))
    return X[feature_order]
# === Predict Route ===
@app.post("/predict")
def predict_churn(data: ChurnInput):
    X_selected = preprocess_input(data)
    prob = model.predict_proba(X_selected)[:, 1][0]
    churn = int(prob > threshold)
    return {'prediction': churn, 'probability': prob}


# === SHAP Explain Route ===
@app.post("/explain")
def explain_churn(data: ChurnInput):
    df = preprocess_input(data)
    input_df = pd.DataFrame(df,columns=feature_order)
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

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

