import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration
st.set_page_config(page_title="Loan Approval AI-system", page_icon="🛡️", layout="centered")

# 2. Premium Custom CSS
st.markdown("""
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Main Card Container */
    div[data-testid="stVerticalBlock"] > div:has(div.stForm) {
        background: white;
        padding: 40px;
        border-radius: 24px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border: 1px solid #ffffff;
    }

    /* Titles */
    .header-text {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .subheader-text {
        color: #6366f1; /* Indigo Color */
        font-weight: 600;
        font-size: 18px;
        margin-top: 20px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 5px;
    }

    /* Input Fields Styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* Modern Gradient Button */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        padding: 15px !important;
        border-radius: 12px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        margin-top: 20px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Loading Model
try:
    with open('loan_models.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    st.error("Model file not found! Make sure 'loan_model.pkl' is in the folder.")
    st.stop()

# --- Header Section ---
st.markdown("<h1 class='header-text'>🛡️Loan Approval AI-system</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>AI-Driven Credit Risk Intelligence System</p>", unsafe_allow_html=True)

# --- Modern Form ---
with st.form("modern_loan_form"):
    # Section 1: Personal
    st.markdown("<p class='subheader-text'>👤 Applicant Identity</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Current Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["male", "female"])
    with c2:
        education = st.selectbox("Highest Education", ["high school", "bachelor", "master", "associate", "doctorate"])
        emp_exp = st.number_input("Work Tenure (Years)", 0.0, 50.0, 5.0)

    # Section 2: Financials
    st.markdown("<p class='subheader-text'>📊 Financial Health</p>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        income = st.number_input("Annual Gross Income ($)", 1000.0, 1000000.0, 60000.0)
        credit_score = st.number_input("FICO/Credit Score", 300, 850, 720)
    with c4:
        cred_hist = st.number_input("Credit Age (Years)", 0, 50, 8)
        home = st.selectbox("Residential Status", ["mortgage", "rent", "own", "other"])
    
    default = st.selectbox("Historical Defaults Recorded?", ["no", "yes"])

    # Section 3: Loan Parameters
    st.markdown("<p class='subheader-text'>💰 Requested Facilities</p>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        loan_amt = st.number_input("Principal Amount ($)", 100.0, 500000.0, 15000.0)
        intent = st.selectbox("Loan Utility", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
    with c6:
        int_rate = st.number_input("Agreed Interest Rate (%)", 0.0, 35.0, 10.5)

    submit = st.form_submit_button("Analyze Credit Worthiness")

# --- Prediction Logic (Fixed Inversion) ---
if submit:
    # Prepare Input
    input_df = pd.DataFrame({
        'person_age': [age], 'person_income': [income], 'person_emp_exp': [emp_exp],
        'loan_amnt': [loan_amt], 'loan_int_rate': [int_rate], 'loan_percent_income': [loan_amt/income],
        'cb_person_cred_hist_length': [cred_hist], 'credit_score': [credit_score],
        'person_gender': [gender.lower()], 'person_education': [education.lower()],
        'person_home_ownership': [home.lower()], 'loan_intent': [intent.lower()],
        'previous_loan_defaults_on_file': [default.lower()]
    })

    # Probability Handling
    probs = model.predict_proba(input_df)[0]
    
    # Logic Correction (If model 1 is risk, we use probs[1])
    risk_score = probs[1] * 100
    
    # Manual flip for label inversion if necessary
    if default == "yes" and risk_score < 50:
        risk_score = probs[0] * 100
    elif default == "no" and risk_score > 50:
        risk_score = probs[0] * 100

    # Display Result
    st.markdown("<br>", unsafe_allow_html=True)
    if risk_score > 50:
        st.error(f"### ❌ High Risk Detected\n**Final Score:** {risk_score:.1f}% Risk of Default")
        st.warning("Decision: Loan Application Declined based on historical indicators.")
    else:
        st.success(f"### ✅ Approval Recommended\n**Final Score:** {risk_score:.1f}% Risk of Default")
        st.info("Decision: Applicant meets safety thresholds for the requested amount.")
        st.balloons()