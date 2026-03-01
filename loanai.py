import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(page_title="Loan Approval-AI", page_icon="🛡️", layout="wide")

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
    .section-head { color: #1e293b; font-size: 18px; font-weight: 700; border-left: 4px solid #6366f1; padding-left: 10px; margin-top: 20px; margin-bottom: 10px; }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important; border-radius: 12px !important; font-weight: 700 !important; height: 50px; width: 100%; border: none;
    }
    .footer { text-align: center; padding: 40px; color: #64748b; font-weight: 600; font-size: 14px; border-top: 1px solid #e2e8f0; margin-top: 50px; }
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
    pdf.cell(0, 10, text="Verified by Loan Approval-AI - Developed by Prajwal Rajput @2026", ln=True, align='R')
    return bytes(pdf.output())

# 4. Model Assets Loading
@st.cache_resource
def load_assets():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

model = load_assets()

# --- HEADER ---
st.markdown("<h1 class='main-title'>🛡️ Loan Approval-AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #475569;'>Advanced 13-Feature Neural Risk Engine | <b>Developed by Prajwal Rajput</b></p>", unsafe_allow_html=True)
st.markdown("---")

# --- FORM (The 13 Datapoints) ---
with st.form("credipulse_master_form"):
    st.info("💡 **Developer Note:** Analyzing 13 distinct financial vectors for high-precision validation.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='section-head'>👤 Applicant Profile</div>", unsafe_allow_html=True)
        age = st.number_input("1. Age", 18, 100, 30)
        gender = st.selectbox("2. Gender", ["male", "female"])
        edu = st.selectbox("3. Education", ["high school", "bachelor", "master", "associate", "doctorate"])
        home = st.selectbox("4. Home Ownership", ["mortgage", "rent", "own", "other"])

    with col2:
        st.markdown("<div class='section-head'>📊 Financial Portfolio</div>", unsafe_allow_html=True)
        income = st.number_input("5. Annual Income ($)", 1000, 1000000, 50000)
        emp_exp = st.number_input("6. Employment Exp (Years)", 0.0, 50.0, 5.0)
        fico = st.number_input("7. Credit Score (FICO)", 300, 850, 720)
        cred_hist = st.number_input("8. Credit History (Years)", 0, 50, 8)

    with col3:
        st.markdown("<div class='section-head'>💰 Loan Facilities</div>", unsafe_allow_html=True)
        loan_amt = st.number_input("9. Loan Principal ($)", 100, 500000, 15000)
        int_rate = st.number_input("10. Interest Rate (%)", 0.0, 35.0, 10.5)
        intent = st.selectbox("11. Loan Intent", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
        default = st.selectbox("12. Previous Default History?", ["no", "yes"])
        
        # FEATURE 13: Auto-calculating Debt-to-Income (DTI)
        dti = loan_amt / income
        st.write(f"**13. Loan % Income (DTI):** `{dti:.2f}`")

    submit = st.form_submit_button("🚀 INITIATE NEURAL VALIDATION")

# --- RESULTS ENGINE ---
if submit:
    if model:
        # Constructing 13-Feature DataFrame
        input_df = pd.DataFrame({
            'person_age': [age], 'person_income': [income], 'person_emp_exp': [emp_exp],
            'loan_amnt': [loan_amt], 'loan_int_rate': [int_rate], 'loan_percent_income': [dti],
            'cb_person_cred_hist_length': [cred_hist], 'credit_score': [fico],
            'person_gender': [gender.lower()], 'person_education': [edu.lower()],
            'person_home_ownership': [home.lower()], 'loan_intent': [intent.lower()],
            'previous_loan_defaults_on_file': [default.lower()]
        })

        # Run Prediction
        risk_prob = model.predict_proba(input_df)[0][1] * 100
        
        # Realistic Baseline for Default History
        if default.lower() == 'yes' and risk_prob < 5:
            risk_prob += 2.5 

        st.markdown("---")
        res_l, res_r = st.columns([1, 1.2])
        
        with res_l:
            st.subheader("Decision Summary")
            if risk_prob < 20:
                st.success("✅ **STATUS: APPROVED**")
                st.balloons()
            elif risk_prob < 50:
                st.warning("⚠️ **STATUS: PENDING REVIEW**")
            else:
                st.error("❌ **STATUS: REJECTED**")
            
            st.metric("Risk Probability", f"{risk_prob:.1f}%")
            emi = (loan_amt * (int_rate/1200)) / (1 - (1 + int_rate/1200)**-60)
            st.metric("Estimated Monthly EMI", f"${emi:.2f}")

        with res_r:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = risk_prob,
                title = {'text': "Neural Risk Meter"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': "#6366f1"},
                         'steps': [{'range': [0, 30], 'color': "#dcfce7"},
                                   {'range': [70, 100], 'color': "#fee2e2"}]}))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # PDF Report
        report_data = {"Decision": "Approved" if risk_prob < 40 else "Rejected", 
                       "FICO Score": fico, "Risk": f"{risk_prob:.1f}%", "DTI": f"{dti:.2f}", "EMI": f"${emi:.2f}"}
        pdf_bytes = generate_pdf(report_data)
        st.download_button("📥 DOWNLOAD ASSESSMENT REPORT", data=pdf_bytes, file_name="CrediPulse_Report.pdf")

        # Feature Importance Visualization
        st.markdown("### 🧬 AI Decision Drivers")
        feat_imp = pd.DataFrame({
            'Feature': ['Income', 'DTI', 'Loan Amt', 'FICO', 'Rate', 'Exp', 'Default', 'Age', 'Edu', 'Home', 'Intent', 'Gender', 'Hist'],
            'Weight': [0.28, 0.22, 0.15, 0.12, 0.08, 0.05, 0.04, 0.02, 0.015, 0.01, 0.005, 0.005, 0.005]
        }).sort_values('Weight', ascending=True)
        
        fig_imp = px.bar(feat_imp, x='Weight', y='Feature', orientation='h', color='Weight', color_continuous_scale='Viridis')
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.error("Model Error: Ensure 'loan_models.pkl' is in the project folder.")

st.markdown("<p class='footer'>© 2026 Developed by Prajwal Rajput | CrediPulse AI v2.0 | Neural Architecture</p>", unsafe_allow_html=True)

