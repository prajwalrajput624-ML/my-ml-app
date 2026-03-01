import streamlit as st
import pandas as pd
import pickle
from fpdf import FPDF
import io

# 1. Page Configuration
st.set_page_config(page_title="Loan Approval AI-system", page_icon="🛡️", layout="centered")

# 2. Custom CSS for Premium Look
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
    </style>
    """, unsafe_allow_html=True)

# 3. PDF Generator Function
def generate_pdf(data_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="🛡️ LOAN ASSESSMENT OFFICIAL REPORT", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for key, value in data_dict.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="Note: This is an AI-generated risk profile for banking reference.", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# 4. Loading Model
@st.cache_resource
def load_loan_model():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

model = load_loan_model()

# --- UI Header ---
st.markdown("<h1 class='header-text'>🛡️ Loan Approval AI-system</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Credit Risk Intelligence & Financial Health Validator</p>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("loan_form"):
    st.markdown("<p class='subheader-text'>👤 Applicant Identity</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Current Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["male", "female"])
        education = st.selectbox("Highest Education", ["high school", "bachelor", "master", "associate", "doctorate"])
    with col2:
        emp_exp = st.number_input("Work Tenure (Years)", 0.0, 50.0, 5.0)
        home = st.selectbox("Residential Status", ["mortgage", "rent", "own", "other"])

    st.markdown("<p class='subheader-text'>📊 Financial Health</p>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        income = st.number_input("Annual Gross Income ($)", 1000.0, 1000000.0, 50000.0)
        credit_score = st.number_input("FICO/Credit Score", 300, 850, 720)
    with col4:
        cred_hist = st.number_input("Credit Age (Years)", 0, 50, 8)
        default = st.selectbox("Historical Defaults?", ["no", "yes"])

    st.markdown("<p class='subheader-text'>💰 Requested Facilities</p>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        loan_amt = st.number_input("Principal Amount ($)", 100.0, 500000.0, 15000.0)
        intent = st.selectbox("Loan Utility", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
    with col6:
        int_rate = st.number_input("Agreed Interest Rate (%)", 0.0, 35.0, 10.5)

    submit = st.form_submit_button("Analyze Credit Worthiness")

# --- Logic & Results ---
if submit:
    if model is None:
        st.error("Model file 'loan_models.pkl' not found!")
    else:
        # 1. Calculations
        dti_ratio = loan_amt / income
        monthly_interest = (int_rate / 100) / 12
        emi = (loan_amt * monthly_interest * (1 + monthly_interest)**60) / ((1 + monthly_interest)**60 - 1)

        # 2. Hard-Rule Safety Check
        if dti_ratio > 0.60 or credit_score < 450:
            st.error("### ❌ Application Denied")
            st.warning(f"High Debt-to-Income Ratio ({dti_ratio:.1%}) or Low Credit Score ({credit_score}).")
        else:
            # 3. Model Inference
            input_data = pd.DataFrame({
                'person_age': [age], 'person_income': [income], 'person_emp_exp': [emp_exp],
                'loan_amnt': [loan_amt], 'loan_int_rate': [int_rate], 'loan_percent_income': [dti_ratio],
                'cb_person_cred_hist_length': [cred_hist], 'credit_score': [credit_score],
                'person_gender': [gender.lower()], 'person_education': [education.lower()],
                'person_home_ownership': [home.lower()], 'loan_intent': [intent.lower()],
                'previous_loan_defaults_on_file': [default.lower()]
            })

            # Get Risk Probability
            risk_prob = model.predict_proba(input_data)[0][1] * 100
            
            # 4. Display Results
            st.markdown("---")
            if risk_prob > 35:
                st.error(f"### ❌ High Risk Detected: {risk_prob:.1f}%")
                st.info("Decision: Based on historical data, this profile shows high volatility.")
            else:
                st.success("### ✅ Approval Recommended")
                st.balloons()
                
                res1, res2, res3 = st.columns(3)
                res1.metric("Risk Score", f"{risk_prob:.1f}%")
                res2.metric("Monthly EMI", f"${emi:.2f}")
                res3.metric("DTI Ratio", f"{dti_ratio:.1%}")

                # 5. Generate PDF
                report_data = {
                    "Status": "APPROVED",
                    "Applicant Age": age,
                    "Annual Income": f"${income:,.2f}",
                    "Loan Amount": f"${loan_amt:,.2f}",
                    "Credit Score": credit_score,
                    "Risk Probability": f"{risk_prob:.1f}%",
                    "Monthly EMI": f"${emi:.2f}"
                }
                pdf_bytes = generate_pdf(report_data)
                
                st.download_button(
                    label="📥 Download Approval Report",
                    data=pdf_bytes,
                    file_name=f"Loan_Report_{credit_score}.pdf",
                    mime="application/pdf"
                )

st.markdown("<br><p style='text-align: center; color: gray; font-size: 12px;'>Standard 60-month tenure assumed. AI prediction based on historical datasets.</p>", unsafe_allow_html=True)
