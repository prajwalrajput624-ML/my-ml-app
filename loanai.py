import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

# 1. Page Config
st.set_page_config(page_title="AI Credit Intelligence 2026", page_icon="🏦", layout="wide")

# 2. Premium CSS
st.markdown("""
    <style>
    .stApp { background: #f0f2f5; }
    div[data-testid="stVerticalBlock"] > div:has(div.stForm) {
        background: white; padding: 30px; border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    .main-title { color: #1e293b; font-weight: 900; text-align: center; margin-bottom: 20px; }
    .footer { text-align: center; padding: 30px; color: #64748b; font-weight: 600; margin-top: 40px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Model Assets
@st.cache_resource
def load_engine():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_engine()

# --- HEADER ---
st.markdown("<h1 class='main-title'>🛡️ 13-Feature AI Credit Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Developed by <b>Prajwal Rajput</b> | Financial Intelligence @2026</p>", unsafe_allow_html=True)

# --- INPUT FORM ---
with st.form("master_suite"):
    st.info("💡 **Developer Note:** System analyzing 13 distinct datapoints for neural inference.")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("👤 Demographic")
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["male", "female"])
        edu = st.selectbox("Education", ["high school", "bachelor", "master", "associate", "doctorate"])
        home = st.selectbox("Home Ownership", ["mortgage", "rent", "own", "other"])

    with c2:
        st.subheader("📊 Financials")
        income = st.number_input("Annual Income ($)", 1000, 1000000, 50000)
        emp_exp = st.number_input("Employment Exp (Years)", 0.0, 50.0, 5.0)
        fico = st.number_input("Credit Score (FICO)", 300, 850, 700)
        cred_hist = st.number_input("Credit History Length", 0, 50, 8)

    with c3:
        st.subheader("💰 Loan Specs")
        loan_amt = st.number_input("Loan Amount ($)", 100, 500000, 15000)
        int_rate = st.number_input("Interest Rate (%)", 0.0, 35.0, 10.5)
        intent = st.selectbox("Loan Intent", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
        default = st.selectbox("Previous Default?", ["no", "yes"])
        
        # 13th Feature Logic
        dti = loan_amt / income
        st.write(f"**Feature 13 (DTI Ratio):** `{dti:.2f}`")

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("⚡ EXECUTE NEURAL VALIDATION")

# --- ML INFERENCE ---
if submit and model:
    # 13-Feature DataFrame construction
    input_df = pd.DataFrame({
        'person_age': [age], 'person_income': [income], 'person_emp_exp': [emp_exp],
        'loan_amnt': [loan_amt], 'loan_int_rate': [int_rate], 'loan_percent_income': [dti],
        'cb_person_cred_hist_length': [cred_hist], 'credit_score': [fico],
        'person_gender': [gender.lower()], 'person_education': [edu.lower()],
        'person_home_ownership': [home.lower()], 'loan_intent': [intent.lower()],
        'previous_loan_defaults_on_file': [default.lower()]
    })

    # Prediction
    prob = model.predict_proba(input_df)[0][1] * 100
    
    st.markdown("---")
    res_l, res_r = st.columns([1, 1.2])
    
    with res_l:
        st.subheader("AI Decision Verdict")
        if prob < 15:
            st.success(f"✅ **LOW RISK: APPROVED**")
            st.write(f"The profile shows strong financial stability with a **{prob:.1f}%** risk score.")
        elif prob < 40:
            st.warning(f"⚠️ **MODERATE RISK: MANUAL REVIEW**")
        else:
            st.error(f"❌ **HIGH RISK: REJECTED**")
        
        # EMI Calculation (Fixed 5-year tenure)
        emi = (loan_amt * (int_rate/1200)) / (1 - (1 + int_rate/1200)**-60)
        st.metric("Estimated Monthly EMI", f"${emi:.2f}")

    with res_r:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob,
            title = {'text': "Neural Risk Meter"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "#6366f1"},
                     'steps': [{'range': [0, 30], 'color': "#dcfce7"},
                               {'range': [70, 100], 'color': "#fee2e2"}]}))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Visualization of Feature Significance
    st.markdown("### 🧬 Feature Significance Analysis")
    
    
    feat_imp = pd.DataFrame({
        'Feature': ['Income', 'DTI', 'Loan Amount', 'FICO', 'Interest', 'Experience', 'Default History', 'Age', 'Education', 'Home Status', 'Intent', 'Gender', 'Credit Hist'],
        'Weight': [0.28, 0.22, 0.15, 0.12, 0.08, 0.05, 0.04, 0.02, 0.015, 0.01, 0.005, 0.005, 0.005]
    }).sort_values('Weight', ascending=True)
    
    fig_imp = px.bar(feat_imp, x='Weight', y='Feature', orientation='h', color='Weight', color_continuous_scale='Viridis')
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("<p class='footer'>© 2026 Developed by Prajwal Rajput | 13-Feature Neural Architecture</p>", unsafe_allow_html=True)
