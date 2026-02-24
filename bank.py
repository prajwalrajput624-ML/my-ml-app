import streamlit as st
import joblib
import pandas as pd

# 1. Model Load karein
model = joblib.load('churn_model_v1.joblib')

st.title("🏦 Bank Customer Churn Predictor")

# 2. User Inputs (Wahi 8 columns jo aapke screenshot mein hain)
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", value=600)
        age = st.number_input("Age", value=30)
        tenure = st.number_input("Tenure", value=5)
        balance = st.number_input("Balance", value=0.0)

    with col2:
        num_products = st.number_input("Number of Products", value=1)
        # 0 aur 1 ko 'No' aur 'Yes' dikhane ke liye fix
        is_active = st.radio("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        salary = st.number_input("Estimated Salary", value=50000.0)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    submit = st.form_submit_button("Predict")

# 3. Prediction Logic (KeyError Fix)
if submit:
    # Aapke screenshot [72] ke mutabiq exact 8 columns ka order:
    input_df = pd.DataFrame([[
        credit_score, age, tenure, balance, 
        num_products, is_active, salary, geography
    ]], columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Geography'])

    # Prediction
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("⚠️ Customer Will Churn")
    else:
        st.success("✅ Customer Will Not Churn")