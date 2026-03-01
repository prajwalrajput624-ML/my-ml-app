import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(page_title="CrediPulse AI | Prajwal Rajput", page_icon="🛡️", layout="wide")

# 2. Ultra-Modern Glassmorphism CSS
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(239, 246, 255) 0%, rgb(219, 234, 254) 100%); }
    div[data-testid="stVerticalBlock"] > div:has(div.stForm) {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(15px);
        padding: 40px; border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 40px rgba(31, 38, 135, 0.1);
    }
    .main-title {
        background: -webkit-linear-gradient(#4f46e5, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px; font-weight: 900; text-align: center; margin-bottom: 0px;
    }
    .section-head { color: #1e293b; font-size: 18px; font-weight: 700; border-left: 4px solid #6366f1; padding-left: 10px; margin-top: 20px; }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important; border-radius: 12px !important; font-weight: 700 !important; height: 50px; width: 100%; border: none;
    }
    .footer { text-align: center; padding: 40px; color: #4b5563; font-size: 14px; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# 3. PDF Generator Function
def generate_pdf(data_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(79, 70, 229) 
    pdf.cell(0, 15, text="CREDIT RISK ASSESSMENT REPORT", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("helvetica", size=12); pdf.set_text_color(30, 41, 59)
    for key, value in data_dict.items():
        clean_line = f"{key}: {value}".encode('ascii', 'ignore').decode('ascii')
        pdf.cell(0, 10, text=clean_line, ln=True)
    pdf.ln(20)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 10, text="Verified by CrediPulse AI - Developed by Prajwal Rajput @2026", ln=True, align='R')
    return bytes(pdf.output())

# 4. Model Loader
@st.cache_resource
def load_engine():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_engine()

# --- HEADER ---
st.markdown("<h1 class='main-title'>🛡️ CrediPulse AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>13-Feature Neural Risk Engine | <b>Developed by Prajwal Rajput</b></p>", unsafe_allow_html=True)

# --- FORM (The 13 Datapoints) ---
with st.form("master_neural_form"):
    st.markdown("<div class='section-head'>👤 Applicant Demographics</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    f1_age = c1.number_input("Age", 18, 100, 30)
    f2_gender = c2.selectbox("Gender", ["male", "female"])
    f3_edu = c3.selectbox("Education", ["high school", "bachelor", "master", "associate", "doctorate"])
    
    st.markdown("<div class='section-head'>📊 Financial Portfolio</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    f4_income = c1.number_input("Annual Income ($)", 1000, 1000000, 55000)
    f5_exp = c2.number_input("Work Experience (Years)", 0.0, 50.0, 5.0)
    f6_home = c3.selectbox("Residential Status", ["mortgage", "rent", "own", "other"])
    
    c1, c2 = st.columns(2)
    f7_fico = c1.number_input("FICO Credit Score", 300, 850, 720)
    f8_cred_hist = c2.number_input("Credit History Length (Years)", 0, 50, 8)
    f9_default = st.radio("Historical Defaults?", ["no", "yes"], horizontal=True)

    st.markdown("<div class='section-head'>💰 Requested Facilities</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    f10_loan_amt = c1.number_input("Loan Principal ($)", 100, 500000, 15000)
    f11_rate = c2.number_input("Interest Rate (%)", 0.0, 35.0, 11.2)
    f12_intent = c3.selectbox("Loan Utility", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
    
    # Feature 13: Auto-Calculation of DTI
    f13_dti = f10_loan_amt / f4_income
    st.write(f"**Feature 13: Debt-to-Income (DTI) Impact:** `{f13_dti:.2f}`")

    submit = st.form_submit_button("⚡ EXECUTE AI VALIDATION")

# --- RESULTS ENGINE ---
if submit and model:
    # Constructing 13-Feature DataFrame
    input_data = pd.DataFrame({
        'person_age': [f1_age], 'person_income': [f4_income], 'person_emp_exp': [f5_exp],
        'loan_amnt': [f10_loan_amt], 'loan_int_rate': [f11_rate], 'loan_percent_income': [f13_dti],
        'cb_person_cred_hist_length': [f8_cred_hist], 'credit_score': [f7_fico],
        'person_gender': [f2_gender.lower()], 'person_education': [f3_edu.lower()],
        'person_home_ownership': [f6_home.lower()], 'loan_intent': [f12_intent.lower()],
        'previous_loan_defaults_on_file': [f9_default.lower()]
    })

    risk_prob = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    res_l, res_r = st.columns([1, 1.2])
    
    with res_l:
        st.subheader("Decision Summary")
        if risk_prob < 15:
            st.success("✅ **APPROVED**")
            st.balloons()
        elif risk_prob < 40:
            st.warning("⚠️ **PENDING REVIEW**")
        else:
            st.error("❌ **REJECTED**")
            
        st.metric("Risk Probability", f"{risk_prob:.1f}%")
        emi = (f10_loan_amt * (f11_rate/1200)) / (1 - (1 + f11_rate/1200)**-60)
        st.metric("Estimated EMI", f"${emi:.2f}")

    with res_r:
        # Professional Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = risk_prob,
            title = {'text': "Neural Risk Meter"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "#6366f1"},
                     'steps': [{'range': [0, 30], 'color': "#dcfce7"},
                               {'range': [70, 100], 'color': "#fee2e2"}]}))
        st.plotly_chart(fig, use_container_width=True)

        report_data = {"Status": "Verified", "Score": f7_fico, "Risk": f"{risk_prob:.1f}%", "DTI": f"{f13_dti:.2f}"}
        pdf = generate_pdf(report_data)
        st.download_button("📥 DOWNLOAD ASSESSMENT REPORT", data=pdf, file_name="CrediPulse_Report.pdf")

st.markdown("<p class='footer'>© 2026 Developed by Prajwal Rajput | 13-Feature Architecture</p>", unsafe_allow_html=True)
