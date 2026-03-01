import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.graph_objects as go

# ================= 1. SYSTEM CONFIG =================
st.set_page_config(page_title="FinGuard AI | Prajwal Rajput", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stButton>button { width: 100%; border-radius: 10px; background: #10a37f; color: white; font-weight: bold; }
    .main-header { text-align: center; color: #10a37f; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. LOAD FULL PIPELINE =================
@st.cache_resource
def load_full_pipeline():
    model_path = 'loan_models.joblib'
    if not os.path.exists(model_path):
        st.error(f"❌ '{model_path}' not found in GitHub!")
        return None
    try:
        # Pura pipeline object load ho raha hai
        model = joblib.load(model_path)
        
        # Check if it's a valid sklearn pipeline
        if hasattr(model, 'predict_proba'):
            return model
        else:
            st.error("❌ Loaded object is NOT a valid Pipeline. Please check your training script.")
            return None
    except Exception as e:
        st.error(f"❌ Load Error: {e}")
        return None

pipeline = load_full_pipeline()

# ================= 3. INDIVIDUAL SCAN UI =================
if pipeline:
    st.markdown("<h1 class='main-header'>🛡️ FinGuard AI Terminal</h1>", unsafe_allow_html=True)
    
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Person Age", 18, 100, 25)
            income = st.number_input("Annual Income ($)", value=50000)
            loan = st.number_input("Loan Amount ($)", value=10000)
            score = st.number_input("Credit Score", 300, 850, 700)
        with col2:
            default = st.selectbox("Previous Defaults?", ["no", "yes"])
            home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
            intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"])
            exp = st.number_input("Experience (Years)", value=5)
        
        submit = st.form_submit_button("🚀 RUN AI VERIFICATION")

    if submit:
        # Model ko wahi 13 columns chahiye jo training mein the
        input_df = pd.DataFrame([{
            'person_age': float(age),
            'person_gender': 'male',
            'person_education': 'Bachelor',
            'person_income': float(income),
            'person_emp_exp': float(exp),
            'person_home_ownership': home,
            'loan_amnt': float(loan),
            'loan_intent': intent,
            'loan_int_rate': 11.0, # Default rate
            'loan_percent_income': float(loan/income),
            'cb_person_cred_hist_length': 5.0,
            'credit_score': float(score),
            'previous_loan_defaults_on_file': default
        }])

        try:
            # Prediction
            # 
            probs = pipeline.predict_proba(input_df)[:, 1]
            raw_prob = probs[0]

            # Logic Flip (Prajwal's Custom Logic)
            if default.lower() == 'yes':
                risk = 1.0 - raw_prob if raw_prob < 0.5 else raw_prob
            else:
                risk = raw_prob if raw_prob < 0.5 else 1.0 - raw_prob

            st.divider()
            if risk < 0.25:
                st.success(f"### ✅ APPROVED (Risk: {risk:.2%})")
                st.balloons()
            else:
                st.error(f"### ❌ REJECTED (Risk: {risk:.2%})")
                
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk*100,
                title={'text': "Risk Percentage"},
                gauge={'bar': {'color': "#10a37f" if risk < 0.25 else "#ff4b4b"}}
            ))
            fig.update_layout(height=250, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Pipeline Execution Error: {e}")
            st.info("Make sure your input column names match exactly with training data.")

st.markdown("<p style='text-align:center;'>Developed by Prajwal Rajput</p>", unsafe_allow_html=True)
