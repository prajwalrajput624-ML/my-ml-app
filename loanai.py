import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(page_title="Loan Approval AI", page_icon="🛡️", layout="wide")

# 2. Premium UI Styling
st.markdown("""
    <style>
    .stApp { background: #f8fafc; }
    div[data-testid="stVerticalBlock"] > div:has(div.stForm) {
        background: white; padding: 40px; border-radius: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;
    }
    .main-title {
        background: -webkit-linear-gradient(#ef4444, #4f46e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px; font-weight: 900; text-align: center;
    }
    .section-head { color: #1e293b; font-size: 18px; font-weight: 700; border-left: 4px solid #6366f1; padding-left: 10px; margin-top: 20px; }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important; border-radius: 12px !important; font-weight: 700 !important; height: 50px;
    }
    .footer { text-align: center; padding: 40px; color: #64748b; font-weight: 600; border-top: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# 3. PDF Generator Function
def generate_pdf(data_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(79, 70, 229) 
    pdf.cell(0, 15, text="LOAN ASSESSMENT REPORT", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("helvetica", size=12); pdf.set_text_color(30, 41, 59)
    for key, value in data_dict.items():
        clean_line = f"{key}: {value}".encode('ascii', 'ignore').decode('ascii')
        pdf.cell(0, 10, text=clean_line, ln=True)
    pdf.ln(20)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 10, text="Verified by Loan Approval-AI Developed by Prajwal Rajput @2026", ln=True, align='R')
    return bytes(pdf.output())

# 4. Assets Loading
@st.cache_resource
def load_assets():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_assets()

# --- HEADER ---
st.markdown("<h1 class='main-title'>🛡️Loan Approval-AI </h1>", unsafe_allow_html=True)

# --- FORM ---
    
    with c1:
        st.markdown("<div class='section-head'>👤 Applicant</div>", unsafe_allow_html=True)
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["male", "female"])
        edu = st.selectbox("Education", ["high school", "bachelor", "master", "associate", "doctorate"])
        home = st.selectbox("Home Ownership", ["mortgage", "rent", "own", "other"])

    with c2:
        st.markdown("<div class='section-head'>📊 Financials</div>", unsafe_allow_html=True)
        income = st.number_input("Annual Income ($)", 1, 1000000, 50000)
        exp = st.number_input("Experience (Years)", 0.0, 50.0, 5.0)
        fico = st.number_input("FICO Score", 300, 850, 720)
        hist = st.number_input("Credit History (Years)", 0, 50, 8)

    with c3:
        st.markdown("<div class='section-head'>💰 Loan Specs</div>", unsafe_allow_html=True)
        loan_amt = st.number_input("Loan Amount ($)", 100, 1000000, 15000)
        rate = st.number_input("Interest Rate (%)", 1.0, 35.0, 10.5)
        intent = st.selectbox("Intent", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
        default = st.selectbox("Past Default?", ["no", "yes"])
        
        dti = loan_amt / income
        st.write(f"**DTI Impact:** `{dti:.2f}`")

    submit = st.form_submit_button("⚡ RUN Loan Approval-AI")

# --- CORE LOGIC (AI + HARD GUARDRAILS) ---
if submit:
    hard_reject = False
    rejection_reasons = []

    if fico < 450:
        hard_reject = True
        rejection_reasons.append("Critical FICO Score (Below 450)")
    if dti > 0.60:
        hard_reject = True
        rejection_reasons.append(f"Unsafe DTI Ratio ({dti:.2f})")
    if income < 5000:
        hard_reject = True
        rejection_reasons.append("Income below minimum threshold")

    if hard_reject:
        st.error("### ❌ STATUS: AUTOMATIC REJECTION")
        for reason in rejection_reasons:
            st.write(f"🔴 **Reason:** {reason}")
        risk_prob = 99.9  
    elif model:
        input_df = pd.DataFrame({
            'person_age': [age], 'person_income': [income], 'person_emp_exp': [exp],
            'loan_amnt': [loan_amt], 'loan_int_rate': [rate], 'loan_percent_income': [dti],
            'cb_person_cred_hist_length': [hist], 'credit_score': [fico],
            'person_gender': [gender.lower()], 'person_education': [edu.lower()],
            'person_home_ownership': [home.lower()], 'loan_intent': [intent.lower()],
            'previous_loan_defaults_on_file': [default.lower()]
        })
        risk_prob = model.predict_proba(input_df)[0][1] * 100
        if default.lower() == 'yes': risk_prob += 5.0

        if risk_prob < 20:
            st.success("✅ **STATUS: APPROVED**")
            st.balloons()
        elif risk_prob < 50:
            st.warning("⚠️ **STATUS: PENDING REVIEW**")
        else:
            st.error("❌ **STATUS: REJECTED**")
    
    # Result Visuals
    res_l, res_r = st.columns([1, 1.2])
    with res_l:
        st.metric("Risk Probability", f"{risk_prob:.1f}%")
        emi = (loan_amt * (rate/1200)) / (1 - (1 + rate/1200)**-60) if income > 0 else 0
        st.metric("Monthly EMI", f"${emi:.2f}")
        
        # --- PDF REPORT BUTTON ---
        report_data = {
            "Decision": "Approved" if risk_prob < 50 else "Rejected",
            "FICO Score": fico,
            "Risk Score": f"{risk_prob:.1f}%",
            "Loan Amount": f"${loan_amt}",
            "Monthly EMI": f"${emi:.2f}",
            "DTI Ratio": f"{dti:.2f}"
        }
        pdf_bytes = generate_pdf(report_data)
        st.download_button("📥 DOWNLOAD ASSESSMENT REPORT", data=pdf_bytes, file_name="Loan_Assessment_Report.pdf")
    
    with res_r:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = risk_prob,
            title = {'text': "Risk Meter"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "#ef4444" if risk_prob > 50 else "#4f46e5"},
                     'steps': [{'range': [0, 30], 'color': "#dcfce7"}, {'range': [70, 100], 'color': "#fee2e2"}]}))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<p class='footer'>© 2026 Developed by Prajwal Rajput | Loan Approval-AI </p>", unsafe_allow_html=True)



