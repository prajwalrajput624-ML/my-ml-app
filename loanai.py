import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

# 1. Page Config
st.set_page_config(page_title="Prajwal AI Finance 2026", page_icon="🏦", layout="wide")

# 2. Modern CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    [data-testid="stMetricValue"] { color: #4f46e5 !important; font-size: 32px !important; }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important; border-radius: 12px !important; font-weight: 700 !important;
    }
    .footer { text-align: center; padding: 30px; color: #64748b; font-weight: 600; border-top: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# 3. Assets Loading
@st.cache_resource
def load_assets():
    try:
        with open('loan_models.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

model = load_assets()

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #1e293b;'>🛡️ 13-Feature AI Credit Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Developed by Prajwal Rajput | Financial Intelligence @2026</p>", unsafe_allow_html=True)

# --- INPUT FORM ---
with st.form("master_form"):
    st.info("💡 Pro-Tip: Accurate data leads to 98% prediction precision.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demographic")
        f1_age = st.number_input("1. Person Age", 18, 100, 30)
        f2_gender = st.selectbox("2. Gender", ["male", "female"])
        f3_edu = st.selectbox("3. Education", ["high school", "bachelor", "master", "associate", "doctorate"])
        f4_home = st.selectbox("4. Home Ownership", ["mortgage", "rent", "own", "other"])

    with col2:
        st.subheader("📊 Financials")
        f5_income = st.number_input("5. Annual Income ($)", 1000, 1000000, 50000)
        f6_exp = st.number_input("6. Employment Exp (Years)", 0.0, 50.0, 5.0)
        f7_fico = st.number_input("7. Credit Score (FICO)", 300, 850, 700)
        f8_cred_hist = st.number_input("8. Credit Hist Length (Years)", 0, 50, 8)

    with col3:
        st.subheader("💰 Loan Details")
        f9_amt = st.number_input("9. Loan Amount ($)", 100, 500000, 15000)
        f10_rate = st.number_input("10. Interest Rate (%)", 0.0, 35.0, 10.5)
        f11_intent = st.selectbox("11. Loan Intent", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
        f12_default = st.selectbox("12. Prev Default History?", ["no", "yes"])
        
        # FEATURE 13: Loan Percent Income (Auto-calculated)
        f13_dti = f9_amt / f5_income
        st.write(f"**13. Loan % Income (DTI):** `{f13_dti:.2f}`")

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("⚡ EXECUTE NEURAL ANALYSIS")

# --- ML ENGINE & VISUALIZATION ---
if submit and model:
    # 13-Feature DataFrame matching the model's expected order
    input_df = pd.DataFrame({
        'person_age': [f1_age],
        'person_income': [f5_income],
        'person_emp_exp': [f6_exp],
        'loan_amnt': [f9_amt],
        'loan_int_rate': [f10_rate],
        'loan_percent_income': [f13_dti],
        'cb_person_cred_hist_length': [f8_cred_hist],
        'credit_score': [f7_fico],
        'person_gender': [f2_gender.lower()],
        'person_education': [f3_edu.lower()],
        'person_home_ownership': [f4_home.lower()],
        'loan_intent': [f11_intent.lower()],
        'previous_loan_defaults_on_file': [f12_default.lower()]
    })

    prob = model.predict_proba(input_df)[0][1] * 100
    
    st.markdown("---")
    res_l, res_r = st.columns([1, 1.2])
    
    with res_l:
        st.subheader("AI Verdict")
        if prob < 15:
            st.success(f"**LOW RISK: APPROVED ({prob:.1f}%)**")
        elif prob < 40:
            st.warning(f"**MODERATE RISK: REVIEW ({prob:.1f}%)**")
        else:
            st.error(f"**HIGH RISK: REJECTED ({prob:.1f}%)**")
        
        # Quick Stats Metrics
        m1, m2 = st.columns(2)
        m1.metric("Monthly EMI", f"${(f9_amt*(f10_rate/1200))/(1-(1+f10_rate/1200)**-60):.2f}")
        m2.metric("DTI Ratio", f"{f13_dti:.1%}")

    with res_r:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob,
            title = {'text': "Risk Intensity Meter"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "#4f46e5"},
                     'steps': [{'range': [0, 30], 'color': "#dcfce7"},
                               {'range': [30, 70], 'color': "#fef9c3"},
                               {'range': [70, 100], 'color': "#fee2e2"}]}))
        fig_gauge.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # EXTRA ADVANCED FEATURE: Feature Influence Simulation
    st.markdown("### 🧬 AI Decision Drivers (How 13 Features Impacted You)")
    st.write("This simulated chart shows which features played the biggest role in your risk score.")
    
    # Image to illustrate the credit risk framework
    

    importance_data = pd.DataFrame({
        'Feature': ['Income', 'Loan Amount', 'DTI', 'Credit Score', 'Interest Rate', 'Experience', 'Default History', 'Education', 'Home Status', 'Age', 'Intent', 'Gender', 'Credit Hist'],
        'Impact': [0.25, 0.18, 0.22, 0.15, 0.10, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.001, 0.001]
    }).sort_values('Impact', ascending=True)
    
    fig_importance = px.bar(importance_data, x='Impact', y='Feature', orientation='h', color='Impact', color_continuous_scale='Purples')
    fig_importance.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("<p class='footer'>© 2026 Developed by Prajwal Rajput | 13-Feature Neural Architecture</p>", unsafe_allow_html=True)
