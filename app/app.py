import streamlit as st
import requests
import shap
import numpy as np
import matplotlib.pyplot as plt

st.title("Credit Card Churn Predictor")

st.markdown("### Customer Information")

with st.form("churn_form"):
    st.markdown("## 🧍 Customer Profile")
    Customer_Age = st.slider("Customer Age", 18, 90, 35)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Dependent_count = st.selectbox("Number of Dependents", list(range(0, 11)), index=2)
    Education_Level = st.selectbox("Education Level", [
        'Uneducated', 'High School', 'College', 'Graduate', 
        'Post-Graduate', 'Doctorate', 'Unknown'
    ])
    Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Income_Category = st.selectbox("Income Category", [
        'Less than $40K', '$40K - $60K', '$60K - $80K',
        '$80K - $120K', '$120K +', 'Unknown'
    ])
    Card_Category = st.selectbox("Card Category", ['Blue', 'Silver', 'Gold', 'Platinum'])

    st.markdown("## 💳 Credit & Usage Metrics")
    Months_on_book = st.slider("Months on Book", 6, 60, 36)
    Total_Relationship_Count = st.slider("Total Relationships with Bank", 1, 6, 3)
    Months_Inactive_12_mon = st.slider("Inactive Months (12M)", 0, 12, 3)
    Contacts_Count_12_mon = st.number_input("Contacts with Bank (12M)", min_value=0, max_value=100, value=3, step=1)

    Credit_Limit = st.number_input("Credit Limit", 500.0, 50000.0, 5000.0)
    Total_Revolving_Bal = st.number_input("Total Revolving Balance", 0.0, 50000.0, 1200.0)
    Avg_Open_To_Buy = st.number_input("Avg Open to Buy", 0.0, 50000.0, 3800.0)
    Avg_Utilization_Ratio = st.slider("Average Utilization Ratio", 0.0, 1.0, 0.25)

    st.markdown("## 💸 Spending Patterns")
    Total_Amt_Chng_Q4_Q1 = st.slider("Spending Amount Change (Q4 → Q1)", 0.0, 3.0, 1.2)
    Total_Trans_Amt = st.number_input("Total Transaction Amount", min_value=0, value=5000)
    Total_Trans_Ct = st.number_input("Total Transaction Count", min_value=0, value=60)
    Total_Ct_Chng_Q4_Q1 = st.slider("Transaction Count Change (Q4 → Q1)", 0.0, 2.0, 0.8)

    submitted = st.form_submit_button("Predict")

if submitted:
    # 💡 Derived Features
    def safe_div(a, b): return a / (b+1) if b else 0

    input_data = {
        # Original base features
        'Customer_Age': Customer_Age,
        'Gender': 1 if Gender == 'Female' else 0,
        'Dependent_count': Dependent_count,
        'Education_Level': Education_Level,
        'Marital_Status': Marital_Status,
        'Income_Category': Income_Category,
        'Card_Category': Card_Category,
        'Months_on_book': Months_on_book,
        'Total_Relationship_Count': Total_Relationship_Count,
        'Months_Inactive_12_mon': Months_Inactive_12_mon,
        'Contacts_Count_12_mon': Contacts_Count_12_mon,
        'Credit_Limit': Credit_Limit,
        'Total_Revolving_Bal': Total_Revolving_Bal,
        'Avg_Open_To_Buy': Avg_Open_To_Buy,
        'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1,
        'Total_Trans_Amt': Total_Trans_Amt,
        'Total_Trans_Ct': Total_Trans_Ct,
        'Total_Ct_Chng_Q4_Q1': Total_Ct_Chng_Q4_Q1,
        'Avg_Utilization_Ratio': Avg_Utilization_Ratio,

        # Derived Features
        'Trans_Amt_per_Trans_Ct': safe_div(Total_Trans_Amt, Total_Trans_Ct),
        'Trans_Amt_per_Rev_Bal': safe_div(Total_Trans_Amt, Total_Revolving_Bal),
        'Trans_Amt_per_Open_To_Buy': safe_div(Total_Trans_Amt, Avg_Open_To_Buy),
        'Trans_Amt_per_Contacts': safe_div(Total_Trans_Amt, Contacts_Count_12_mon),
        'Trans_Amt_per_Inactive': safe_div(Total_Trans_Amt, Months_Inactive_12_mon),
        'Trans_Amt_per_Relationship': safe_div(Total_Trans_Amt, Total_Relationship_Count),
        'Trans_Amt_per_Dependent': safe_div(Total_Trans_Amt, Dependent_count),
        'Rev_Bal_per_Contacts': safe_div(Total_Revolving_Bal, Contacts_Count_12_mon),
        'Rev_Bal_per_Relationship': safe_div(Total_Revolving_Bal, Total_Relationship_Count),
        'Credit_Limit_per_Trans_Ct': safe_div(Credit_Limit, Total_Trans_Ct),
        'Credit_Limit_per_Inactive': safe_div(Credit_Limit, Months_Inactive_12_mon),
        'Credit_Limit_per_Relationship': safe_div(Credit_Limit, Total_Relationship_Count),
        'Credit_Limit_per_Dependent': safe_div(Credit_Limit, Dependent_count),
        'Amt_Chng_per_Trans_Ct': safe_div(Total_Amt_Chng_Q4_Q1, Total_Trans_Ct),
        'Amt_Chng_per_Contacts': safe_div(Total_Amt_Chng_Q4_Q1, Contacts_Count_12_mon),
        'Amt_Chng_per_Inactive': safe_div(Total_Amt_Chng_Q4_Q1, Months_Inactive_12_mon),
        'Amt_Chng_per_Relationship': safe_div(Total_Amt_Chng_Q4_Q1, Total_Relationship_Count),
        'Amt_Chng_per_Dependent': safe_div(Total_Amt_Chng_Q4_Q1, Dependent_count),
        'Trans_Ct_per_Dependent': safe_div(Total_Trans_Ct, Dependent_count),
        'Trans_Ct_per_Months_Inactive': safe_div(Total_Trans_Ct, Months_Inactive_12_mon),
        'Trans_Ct_per_Relationship': safe_div(Total_Trans_Ct, Total_Relationship_Count),
        'Sum_Trans_Amt_Rev_Bal': Total_Trans_Amt + Total_Revolving_Bal,
        'Sum_Trans_Amt_Open_To_Buy': Total_Trans_Amt + Avg_Open_To_Buy
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=input_data)
        result = response.json()
        prob = result["probability"]

        st.metric("Churn Probability", f"{prob*100:.2f}%")
        if prob > 0.7:
            st.error("⚠️ High risk of churn. Suggest immediate retention offer.")
        elif prob > 0.5:
            st.warning("🟠 Moderate churn risk. Consider soft retention incentives.")
        else:
            st.success("🟢 Low churn risk. No immediate action needed.")
            
        exp_response = requests.post("http://localhost:8000/explain", json=input_data)

        if exp_response.status_code != 200:
            st.error(f"SHAP explanation failed: {exp_response.text}")
        else:
            exp_result = exp_response.json()

            # Extracting parts from JSON
            base_value = exp_result["base_value"]
            shap_values = np.array(exp_result["shap_values"])  # shape (n_features,)
            feature_values = np.array(exp_result["feature_values"])  # shape (n_features,)
            feature_names = exp_result["feature_names"]  # list of str

            # Build SHAP Explanation for a single prediction
            expl = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=feature_values,
                feature_names=feature_names
            )

            # Plot using modern API
            st.subheader("🔍 SHAP Feature Impact (Waterfall)")
            shap.plots.waterfall(expl, max_display=15, show=False)  # Updated function
            fig = plt.gcf()
            st.pyplot(fig)

            # Display Top Contributors
            st.markdown("#### 📊 Top 5 Influential Features")

            for feature, contribution in exp_result["top_contributors"]:
                symbol = "⬆️" if contribution > 0 else "⬇️"
                st.write(f"{symbol} **{feature}**")





    except Exception as e:
        st.error(f"Error: {str(e)} — is your FastAPI backend running?")
