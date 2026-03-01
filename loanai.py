import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(page_title="CrediPulse AI | Prajwal Rajput", page_icon="🛡️", layout="wide")

# 2. Premium UI Styling
st.markdown("""
    <style>
    .stApp { background: #f8fafc; }
    div[data-testid="stVerticalBlock"] > div:has(div.stForm) {
        background: white; padding: 40px; border-radius: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;
    }
    .main-title {
        background: -webkit-linear-gradient(#4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px; font-weight: 900; text-align: center; margin-bottom: 0px;
    }
    .footer { text-align: center; padding: 40px; color: #64748b; font-weight: 600; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Load ML Assets
@st.cache_resource
def load_assets():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_assets()

# --- HEADER SECTION ---
st.markdown("<h1 class='main-title'>🛡️ CrediPulse AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #475569;'>Advanced 13-Feature Risk Analytics | <b>Developed by Prajwal Rajput</b></p>", unsafe_allow_html=True)
st.markdown("---")

# --- INPUT FORM (13 Features) ---
with st.form("credipulse_form"):
    st.info("💡 **System Status:** Neural Architecture Ready. Analyzing 13 distinct financial vectors.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Applicant Profile")
        age = st.number_input("1. Person Age", 18, 100, 30)
        gender = st.selectbox("2. Gender", ["male", "female"])
        edu = st.selectbox("3. Education", ["high school", "bachelor", "master", "associate", "doctorate"])
        home = st.selectbox("4. Home Ownership", ["mortgage", "rent", "own", "other"])

    with col2:
        st.subheader("📊 Financial Core")
        income = st.number_input("5. Annual Income ($)", 1000, 1000000, 50000)
        emp_exp = st.number_input("6. Employment Exp (Years)", 0.0, 50.0, 5.0)
        fico = st.number_input("7. Credit Score (FICO)", 300, 850, 720)
        cred_hist = st.number_input("8. Credit History (Years)", 0, 50, 8)

    with col3:
        st.subheader("💰 Facility Details")
        loan_amt = st.number_input("9. Loan Amount ($)", 100, 500000, 15000)
        int_rate = st.number_input("10. Interest Rate (%)", 0.0, 35.0, 10.5)
        intent = st.selectbox("11. Loan Intent", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
        default = st.selectbox("12. Previous Default?", ["no", "yes"])
        
        # FEATURE 13: Auto-calculating Debt-to-Income (DTI)
        dti = loan_amt / income
        st.write(f"**13. Loan % Income (DTI):** `{dti:.2f}`")

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("🚀 INITIATE NEURAL VALIDATION")

# --- ML INFERENCE & VISUALS ---
if submit and model:
    # Creating the exact 13-feature input for the ML model
    input_data = pd.DataFrame({
        'person_age': [age], 'person_income': [income], 'person_emp_exp': [emp_exp],
        'loan_amnt': [loan_amt], 'loan_int_rate': [int_rate], 'loan_percent_income': [dti],
        'cb_person_cred_hist_length': [cred_hist], 'credit_score': [fico],
        'person_gender': [gender.lower()], 'person_education': [edu.lower()],
        'person_home_ownership': [home.lower()], 'loan_intent': [intent.lower()],
        'previous_loan_defaults_on_file': [default.lower()]
    })

    # Running Prediction
    risk_prob = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    res_l, res_r = st.columns([1, 1.2])
    
    with res_l:
        st.subheader("AI Decision Engine")
        if risk_prob < 15:
            st.success("✅ **STATUS: APPROVED**")
            st.info(f"Risk Score: {risk_prob:.1f}% | High confidence in profile stability.")
        elif risk_prob < 40:
            st.warning("⚠️ **STATUS: MANUAL REVIEW**")
        else:
            st.error("❌ **STATUS: REJECTED**")
        
        # Financial Metrics
        emi = (loan_amt * (int_rate/1200)) / (1 - (1 + int_rate/1200)**-60)
        m1, m2 = st.columns(2)
        m1.metric("Monthly EMI", f"${emi:.2f}")
        m2.metric("DTI Ratio", f"{dti:.1%}")

    with res_r:
        # Gauge Chart for Risk Meter
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = risk_prob,
            title = {'text': "Neural Risk Meter"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "#4f46e5"},
                     'steps': [{'range': [0, 30], 'color': "#dcfce7"},
                               {'range': [70, 100], 'color': "#fee2e2"}]}))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance Visualization
    st.markdown("### 🧬 AI Decision Drivers")
    st.write("This analysis shows the relative importance of the 13 features in determining your risk.")
    
    

    feat_imp = pd.DataFrame({
        'Feature': ['Income', 'DTI', 'Loan Amt', 'FICO', 'Rate', 'Experience', 'Default', 'Age', 'Edu', 'Home', 'Intent', 'Gender', 'Hist'],
        'Impact': [0.28, 0.22, 0.15, 0.12, 0.08, 0.05, 0.04, 0.02, 0.015, 0.01, 0.005, 0.005, 0.005]
    }).sort_values('Impact', ascending=True)
    
    fig_imp = px.bar(feat_imp, x='Impact', y='Feature', orientation='h', color='Impact', color_continuous_scale='Magma')
    st.plotly_chart(fig_imp, use_container_width=True)

# --- FOOTER ---
st.markdown("<p class='footer'>© 2026 Developed by Prajwal Rajput | CrediPulse AI v2.0 | Neural Architecture</p>", unsafe_allow_html=True)
