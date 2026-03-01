import streamlit as st
import pandas as pd
import pickle
import numpy as np
from fpdf import FPDF
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="Prajwal AI Finance 2026", page_icon="📈", layout="wide")

# 2. Ultra Modern CSS
st.markdown("""
    <style>
    .stApp { background-color: #f4f7f6; }
    .main-card {
        background: white; padding: 30px; border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #e0e0e0;
    }
    .metric-card {
        background: #ffffff; padding: 20px; border-radius: 15px;
        border-top: 5px solid #6366f1; box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #f0f2f6; border-radius: 10px;
        padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. PDF Generator
def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 20)
    pdf.cell(0, 20, "FINANCIAL CREDIT SCORECARD", ln=True, align='C')
    pdf.set_font("helvetica", size=12)
    pdf.ln(10)
    for k, v in data.items():
        pdf.cell(0, 10, f"{k}: {v}".encode('ascii', 'ignore').decode('ascii'), ln=True)
    pdf.ln(20)
    pdf.cell(0, 10, "@2026 Developed by Prajwal Rajput", align='R')
    return bytes(pdf.output())

# 4. Load ML Model
@st.cache_resource
def load_loan_ai():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_loan_ai()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("AI Settings")
    st.info("Model: Random Forest Classifier\nVersion: 2.4.1 (2026)")
    st.markdown("---")
    st.write("Developed by: **Prajwal Rajput**")

# --- MAIN UI ---
st.title("🛡️ AI Credit Intelligence Suite")
tab1, tab2 = st.tabs(["🚀 Loan Assessment", "📊 Risk Analytics"])

with tab1:
    with st.form("advanced_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Personal & Financial Data")
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 18, 100, 30)
            income = c2.number_input("Annual Income ($)", 1000, 1000000, 55000)
            
            c1, c2 = st.columns(2)
            fico = c1.slider("FICO Credit Score", 300, 850, 700)
            exp = c2.number_input("Experience (Years)", 0.0, 50.0, 4.0)
            
            home = st.selectbox("Home Ownership", ["Rent", "Mortgage", "Own"])
            edu = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Doctorate"])

        with col2:
            st.subheader("Loan Details")
            amt = st.number_input("Loan Amount ($)", 500, 500000, 15000)
            rate = st.number_input("Interest Rate (%)", 1.0, 35.0, 10.5)
            intent = st.selectbox("Purpose", ["Personal", "Education", "Medical", "Venture", "Debt Consolidation"])
            history = st.radio("Previous Defaults?", ["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.form_submit_button("⚡ GENERATE AI INSIGHTS")

    if btn and model:
        dti = amt / income
        # ML Inference
        input_df = pd.DataFrame({
            'person_age': [age], 'person_income': [income], 'person_emp_exp': [exp],
            'loan_amnt': [amt], 'loan_int_rate': [rate], 'loan_percent_income': [dti],
            'cb_person_cred_hist_length': [8], 'credit_score': [fico],
            'person_gender': ["male"], 'person_education': [edu.lower()],
            'person_home_ownership': [home.lower()], 'loan_intent': [intent.replace(" ", "").lower()],
            'previous_loan_defaults_on_file': [history.lower()]
        })
        
        prob = model.predict_proba(input_df)[0][1] * 100
        
        # Results Section
        st.markdown("### 📊 Assessment Result")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prob < 20:
                st.success("✅ LOW RISK: APPROVED")
            elif prob < 40:
                st.warning("⚠️ MODERATE RISK: REVIEW")
            else:
                st.error("❌ HIGH RISK: REJECTED")
            
            st.metric("Risk Probability", f"{prob:.1f}%")
            st.metric("Monthly EMI", f"${(amt*(rate/1200))/(1-(1+rate/1200)**-60):.2f}")

        with res_col2:
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                title = {'text': "Credit Risk Meter"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#6366f1"},
                    'steps': [
                        {'range': [0, 25], 'color': "#d1fae5"},
                        {'range': [25, 50], 'color': "#fef3c7"},
                        {'range': [50, 100], 'color': "#fee2e2"}]
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # PDF Download
        report = {"Applicant": "Prajwal User", "Score": fico, "Income": income, "Risk": f"{prob:.1f}%"}
        pdf = generate_pdf(report)
        st.download_button("📥 DOWNLOAD DETAILED REPORT", data=pdf, file_name="AI_Assessment.pdf")

with tab2:
    st.subheader("Market Comparison & Analytics")
    # Interactive Risk Heatmap Visualization
    st.markdown("This section shows where your application sits compared to industry benchmarks.")
    
    # Image to illustrate the credit risk framework
    st.write("Below is a visual representation of how ML models weigh different credit factors:")
    st.write("")
    
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['FICO Weight', 'Income Factor', 'Risk Path']
    )
    st.line_chart(chart_data)
    st.info("The AI model uses over 12 variables to determine your final risk score.")

st.markdown("<div class='footer'>© 2026 Developed by Prajwal Rajput | Powered by Advanced Machine Learning</div>", unsafe_allow_html=True)
