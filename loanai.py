import streamlit as st
import pandas as pd
import pickle
from fpdf import FPDF
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Prajwal AI Finance 2026", page_icon="💳", layout="wide")

# 2. Ultra-Modern CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-card { background: white; padding: 30px; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important; border-radius: 12px !important; font-weight: 700 !important; height: 50px; width: 100%;
    }
    .footer { text-align: center; padding: 30px; color: #64748b; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# 3. Model & Scaler Loading
@st.cache_resource
def load_assets():
    try:
        # Maan ke chal rahe hain ki 'loan_models.pkl' mein trained model hai
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_assets()

# --- SIDEBAR (Model Info) ---
with st.sidebar:
    st.title("⚙️ AI Core Settings")
    st.info("Status: Neural Engine Active\nInput Features: 13\nTarget: Default Probability")
    st.markdown("---")
    st.write("Developed by: **Prajwal Rajput**")

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #1e293b;'>🛡️ Advanced Credit Risk Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Full-Feature ML Validation @2026</p>", unsafe_allow_html=True)

# --- INPUT FORM (Handling all 13 Features) ---
with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Identity")
        age = st.number_input("Person Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["male", "female"])
        edu = st.selectbox("Education", ["high school", "bachelor", "master", "associate", "doctorate"])
        
    with col2:
        st.markdown("### 📊 Financials")
        income = st.number_input("Annual Income ($)", 1000, 1000000, 50000)
        emp_exp = st.number_input("Employment Exp (Years)", 0.0, 50.0, 5.0)
        home = st.selectbox("Home Ownership", ["mortgage", "rent", "own", "other"])
        cred_hist = st.number_input("Credit History Length", 0, 50, 8)

    with col3:
        st.markdown("### 💰 Loan Details")
        loan_amt = st.number_input("Loan Amount ($)", 100, 500000, 15000)
        int_rate = st.number_input("Interest Rate (%)", 0.0, 35.0, 10.5)
        intent = st.selectbox("Loan Intent", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
        fico = st.number_input("Credit Score (FICO)", 300, 850, 700)
        default_file = st.selectbox("Previous Default?", ["no", "yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("⚡ EXECUTE 13-FEATURE ANALYSIS")

# --- RESULTS & ML INFERENCE ---
if submit and model:
    # Feature 13: Loan Percent Income (DTI) - Automatic Calculation
    loan_percent_income = loan_amt / income
    
    # Building the exact 13-feature DataFrame
    input_df = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_emp_exp': [emp_exp],
        'loan_amnt': [loan_amt],
        'loan_int_rate': [int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cred_hist],
        'credit_score': [fico],
        'person_gender': [gender.lower()],
        'person_education': [edu.lower()],
        'person_home_ownership': [home.lower()],
        'loan_intent': [intent.lower()],
        'previous_loan_defaults_on_file': [default_file.lower()]
    })

    # ML Prediction
    risk_prob = model.predict_proba(input_df)[0][1] * 100
    
    st.markdown("---")
    res_l, res_r = st.columns([1, 1])
    
    with res_l:
        st.subheader("AI Decision Verdict")
        if risk_prob < 15:
            st.success(f"**LOW RISK ({risk_prob:.1f}%)**\n\nDecision: Instant Approval Recommended.")
        elif risk_prob < 40:
            st.warning(f"**MODERATE RISK ({risk_prob:.1f}%)**\n\nDecision: Manual Underwriting Required.")
        else:
            st.error(f"**HIGH RISK ({risk_prob:.1f}%)**\n\nDecision: Application Rejected.")

    with res_r:
        # Professional Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob,
            title = {'text': "Risk Intensity Index"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "#4f46e5"},
                     'steps': [
                         {'range': [0, 30], 'color': "#dcfce7"},
                         {'range': [30, 70], 'color': "#fef9c3"},
                         {'range': [70, 100], 'color': "#fee2e2"}]}))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<p class='footer'>© 2026 Developed by Prajwal Rajput | Full 13-Feature Neural Engine</p>", unsafe_allow_html=True)
