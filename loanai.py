import streamlit as st
import pandas as pd
import pickle
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(page_title="AI Credit Intelligence", page_icon="💳", layout="wide")

# 2. Advanced Custom CSS
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(239, 246, 255) 0%, rgb(219, 234, 254) 100%); }
    div[data-testid="stVerticalBlock"] > div:has(div.stForm) {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(15px);
        padding: 50px; border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    }
    .main-title {
        background: -webkit-linear-gradient(#4f46e5, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 45px; font-weight: 900; text-align: center;
    }
    .section-head { color: #1e293b; font-size: 20px; font-weight: 700; margin-top: 30px; }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important; border-radius: 15px !important;
        font-weight: 700 !important; width: 100%; height: 50px;
    }
    .footer { text-align: center; padding: 40px; color: #6366f1; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# 3. PDF Generator (Fixed NameError)
def generate_pdf(data_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(79, 70, 229) 
    pdf.cell(0, 15, text="CREDIT RISK ASSESSMENT REPORT", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("helvetica", size=12)
    pdf.set_text_color(30, 41, 59)
    for key, value in data_dict.items():
        # FIXED: Variable name consistency
        clean_line = f"{key}: {value}".encode('ascii', 'ignore').decode('ascii')
        pdf.cell(0, 10, text=clean_line, ln=True)
    
    pdf.ln(20)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 10, text="Developed by Prajwal Rajput AI-system", ln=True, align='R')
    return bytes(pdf.output())

# 4. Model Loader
@st.cache_resource
def load_model():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_model()

# --- UI Layout ---
st.markdown("<h1 class='main-title'>🛡️ AI Loan Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Developed by Prajwal Rajput</p>", unsafe_allow_html=True)

with st.form("modern_form"):
    st.markdown("<div class='section-head'>👤 Applicant Profile</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Current Age", 18, 100, 30)
    gender = c2.selectbox("Gender", ["Male", "Female"])
    edu = c3.selectbox("Education", ["High School", "Bachelor", "Master", "Associate", "Doctorate"])
    
    c1, c2 = st.columns(2)
    exp = c1.number_input("Work Tenure (Years)", 0.0, 50.0, 5.0)
    home = c2.selectbox("Residential Status", ["Mortgage", "Rent", "Own", "Other"])

    st.markdown("<div class='section-head'>📊 Financial Intelligence</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    income = c1.number_input("Annual Income ($)", 1000, 1000000, 50000)
    fico = c2.number_input("FICO Credit Score", 300, 850, 720)
    cred_age = c3.number_input("Credit Age (Years)", 0, 50, 8)
    default_his = st.radio("Have you defaulted in the past?", ["No", "Yes"], horizontal=True)

    st.markdown("<div class='section-head'>💰 Loan Requirements</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    loan_amt = c1.number_input("Requested Amount ($)", 100, 500000, 15000)
    intent = c2.selectbox("Purpose", ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Debt Consolidation"])
    rate = c3.number_input("Interest Rate (%)", 0.0, 35.0, 10.5)

    submit = st.form_submit_button("⚡ RUN AI VALIDATION")

# --- Processing ---
if submit and model:
    dti = loan_amt / income
    monthly_int = (rate / 100) / 12
    emi = (loan_amt * monthly_int * (1 + monthly_int)**60) / ((1 + monthly_int)**60 - 1)

    if dti > 0.60 or fico < 450:
        st.error("### ❌ Risk Threshold Exceeded")
    else:
        input_data = pd.DataFrame({
            'person_age': [age], 'person_income': [income], 'person_emp_exp': [exp],
            'loan_amnt': [loan_amt], 'loan_int_rate': [rate], 'loan_percent_income': [dti],
            'cb_person_cred_hist_length': [cred_age], 'credit_score': [fico],
            'person_gender': [gender.lower()], 'person_education': [edu.lower()],
            'person_home_ownership': [home.lower()], 'loan_intent': [intent.replace(" ", "").lower()],
            'previous_loan_defaults_on_file': [default_his.lower()]
        })
        
        prob = model.predict_proba(input_data)[0][1] * 100
        
        st.markdown("---")
        if prob > 35:
            st.error(f"### ❌ High Default Probability: {prob:.1f}%")
        else:
            st.success("### ✅ Application Approved by AI")
            st.progress(int(prob), text=f"Risk Intensity: {prob:.1f}%")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Risk Score", f"{prob:.1f}%")
            m2.metric("Monthly EMI", f"${emi:.2f}")
            m3.metric("DTI Ratio", f"{dti:.1%}")
            m4.metric("FICO Status", "Prime" if fico > 700 else "Subprime")

            rep_data = {"Status": "Approved", "FICO": fico, "Risk": f"{prob:.1f}%", "EMI": f"${emi:.2f}"}
            pdf_out = generate_pdf(rep_data)
            # FIXED: file_name as string
            st.download_button("📥 DOWNLOAD REPORT", data=pdf_out, file_name="Loan_Assessment_Report.pdf")
            st.balloons()

st.markdown("<p class='footer'>DEVELOPED BY PRAJWAL RAJPUT</p>", unsafe_allow_html=True)
