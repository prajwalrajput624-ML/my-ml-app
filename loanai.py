import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Loan Approval AI-system", page_icon="🛡️", layout="centered")

# 2. Premium Custom CSS
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
    div[data-testid="stVerticalBlock"] > div:has(div.stForm) {
        background: white; padding: 40px; border-radius: 24px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); border: 1px solid #ffffff;
    }
    .header-text { color: #1e293b; font-weight: 800; text-align: center; margin-bottom: 0px; }
    .subheader-text { color: #6366f1; font-weight: 600; font-size: 18px; margin-top: 20px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important; width: 100%; border-radius: 12px !important; font-weight: 700 !important; height: 50px;
    }
    .metric-card { background: #f1f5f9; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# 3. Loading Model Function
@st.cache_resource
def load_model():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# --- Header ---
st.markdown("<h1 class='header-text'>🛡️ Loan Approval AI-system</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Credit Risk Intelligence & Financial Health Validator</p>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("modern_loan_form"):
    st.markdown("<p class='subheader-text'>👤 Applicant Identity</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Current Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["male", "female"])
    with c2:
        education = st.selectbox("Highest Education", ["high school", "bachelor", "master", "associate", "doctorate"])
        emp_exp = st.number_input("Work Tenure (Years)", 0.0, 50.0, 5.0)

    st.markdown("<p class='subheader-text'>📊 Financial Health</p>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        income = st.number_input("Annual Gross Income ($)", 1000.0, 1000000.0, 50000.0)
        credit_score = st.number_input("FICO/Credit Score", 300, 850, 720)
    with c4:
        cred_hist = st.number_input("Credit Age (Years)", 0, 50, 8)
        home = st.selectbox("Residential Status", ["mortgage", "rent", "own", "other"])
    
    default = st.selectbox("Historical Defaults Recorded?", ["no", "yes"])

    st.markdown("<p class='subheader-text'>💰 Requested Facilities</p>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        loan_amt = st.number_input("Principal Amount ($)", 100.0, 500000.0, 15000.0)
        intent = st.selectbox("Loan Utility", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
    with c6:
        int_rate = st.number_input("Agreed Interest Rate (%)", 0.0, 35.0, 10.5)

    submit = st.form_submit_button("Analyze Credit Worthiness")

# --- Logic Processing ---
if submit:
    if model is None:
        st.error("Model file not found! Please check 'loan_models.pkl'.")
    else:
        # 1. Calculations
        loan_percent_income = loan_amt / income
        monthly_rate = (int_rate / 100) / 12
        # Assuming a standard 5-year (60 months) term for EMI calculation
        tenure_months = 60 
        emi = (loan_amt * monthly_rate * (1 + monthly_rate)**tenure_months) / ((1 + monthly_rate)**tenure_months - 1)

        # 2. Hard-Stop Validation (Pre-Model)
        # Rejection if Loan is > 60% of income OR Credit Score is too low
        if loan_percent_income > 0.60 or credit_score < 450:
            st.markdown("---")
            st.error("### ❌ Application Denied")
            st.warning(f"**Reason:** High Debt-to-Income Ratio ({loan_percent_income:.1%}) or Low Credit Score ({credit_score}).")
            st.info("Financial logic suggests this loan exceeds repayment capacity.")
        
        else:
            # 3. Model Inference
            input_df = pd.DataFrame({
                'person_age': [age], 'person_income': [income], 'person_emp_exp': [emp_exp],
                'loan_amnt': [loan_amt], 'loan_int_rate': [int_rate], 'loan_percent_income': [loan_percent_income],
                'cb_person_cred_hist_length': [cred_hist], 'credit_score': [credit_score],
                'person_gender': [gender.lower()], 'person_education': [education.lower()],
                'person_home_ownership': [home.lower()], 'loan_intent': [intent.lower()],
                'previous_loan_defaults_on_file': [default.lower()]
            })

            # Prediction
            probs = model.predict_proba(input_df)[0]
            risk_score = probs[1] * 100  # Probability of Default

            st.markdown("---")
            
            # 4. Results Display
            if risk_score > 35:
                st.error(f"### ❌ High Risk Detected")
                st.metric("Default Probability", f"{risk_score:.1f}%")
                st.write("**Decision:** Application Declined based on AI risk assessment.")
            else:
                st.success(f"### ✅ Approval Recommended")
                st.balloons()
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Risk Score", f"{risk_score:.1f}%")
                with res_col2:
                    st.metric("Est. Monthly EMI", f"${emi:.2f}")
                
                st.info(f"**Decision:** Applicant meets safety thresholds for ${loan_amt:,.2f}.")

st.markdown("<br><p style='text-align: center; color: gray; font-size: 12px;'>Standard 60-month tenure assumed for EMI calculations.</p>", unsafe_allow_html=True)
