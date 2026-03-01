import streamlit as st
import pandas as pd
import joblib  # Joblib load karne ke liye zaroori hai
import time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# ================= 1. SYSTEM CONFIG =================
pd.set_option("styler.render.max_elements", 1500000)
st.set_page_config(page_title="Loan Approval AI | Prajwal Rajput", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stButton>button { width: 100%; border-radius: 10px; background: #10a37f; color: white; font-weight: bold; height: 3em; }
    .main-header { text-align: center; color: #10a37f; margin-bottom: 20px; font-family: 'Trebuchet MS'; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #161b22; color: #8b949e; text-align: center;
        padding: 10px; border-top: 1px solid #30363d; z-index: 100;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. LOGIN SYSTEM =================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h1 class='main-header'>🛡️ Secure Login</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1.5,1])
    with col2:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Launch Dashboard"):
                if u == "prajwal" and p == "prajwal6575":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
    st.stop()

# ================= 3. JOBLIB MODEL LOAD (THE FIX) =================
@st.cache_resource
def load_joblib_model():
    # File ka naam check karein: 'loan_model.joblib' ya 'loan_model.pkl'
    model_file = 'loan_models.joblib' 
    if not os.path.exists(model_file):
        st.error(f"❌ File '{model_file}' not found in your repository!")
        return None
    try:
        # Joblib loading
        model_obj = joblib.load(model_file)
        
        # Verify agar ye real model hai ya sirf string
        if hasattr(model_obj, "predict_proba"):
            return model_obj
        else:
            st.error("❌ The file loaded as a string. Please re-export your model using joblib.dump()")
            return None
    except Exception as e:
        st.error(f"❌ Error loading Joblib file: {e}")
        return None

pipeline = load_joblib_model()
if pipeline is None:
    st.stop()

# ================= 4. SIDEBAR =================
st.sidebar.write(f"Logged in: **Prajwal Rajput**")
mode = st.sidebar.radio("Navigation", ["Individual Scan", "Upload csv"])
if st.sidebar.button("🔒 Logout"):
    st.session_state.authenticated = False
    st.rerun()

# ================= 5. MODE 1: INDIVIDUAL SCAN =================
if mode == "Individual Scan":
    st.markdown("<h2 class='main-header'>Individual Profile Analysis</h2>", unsafe_allow_html=True)
    with st.form("single_entry"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 100, 25)
            income = st.number_input("Annual Income ($)", value=45000)
            loan = st.number_input("Loan Amount ($)", value=12000)
            score = st.number_input("Credit Score", 300, 850, 710)
        with c2:
            default = st.selectbox("Previous Default", ["no", "yes"])
            home = st.selectbox("Home Ownership", ["rent", "own", "mortgage"])
            intent = st.selectbox("Loan Purpose", ["personal", "education", "medical", "venture"])
            exp = st.number_input("Job Experience (Years)", value=5)
        btn = st.form_submit_button("🚀 Running Loan-Approval AI")

    if btn:
        with st.spinner("AI is analyzing..."):
            # Model expectation: 13 features in EXACT order
            full_data = pd.DataFrame([{
                'person_age': age, 
                'person_gender': 'male', 
                'person_education': 'bachelor',
                'person_income': float(income), 
                'person_emp_exp': float(exp),
                'person_home_ownership': home.lower(), 
                'loan_amnt': float(loan), 
                'loan_intent': intent.lower(), 
                'loan_int_rate': 11.0, # Default rate agar user input nahi hai
                'loan_percent_income': float(loan/income),
                'cb_person_cred_hist_length': 5.0, 
                'credit_score': float(score),
                'previous_loan_defaults_on_file': default.lower()
            }])
            
            try:
                # Prediction yahan fail ho rahi thi
                raw_prob = pipeline.predict_proba(full_data)[0][1]
                
                # Risk Logic Flip
                if default == 'yes':
                    risk = 1.0 - raw_prob if raw_prob < 0.5 else raw_prob
                else:
                    risk = raw_prob if raw_prob < 0.5 else 1.0 - raw_prob
                
                st.divider()
                res_c1, res_c2 = st.columns([1.5, 1])
                with res_c1:
                    if risk < 0.25:
                        st.success(f"### ✅ APPROVED | Confidence: {100-(risk*100):.2f}%")
                        st.balloons()
                    else:
                        st.error(f"### ❌ REJECTED | Risk: {risk:.2%}")
                with res_c2:
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=risk*100, 
                                               gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#10a37f"}}))
                    fig.update_layout(height=230, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Pipeline Error: {e}")
                st.info("Check if your Joblib file contains the full Pipeline (Preprocessing + Model).")

# ================= 6. MODE 2: BATCH SCAN =================
else:
    st.markdown("<h2 class='main-header'>Bulk Applicant Processing</h2>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        if st.button("⚡ EXECUTE MASSIVE SCAN"):
            try:
                # Bulk prediction using the loaded joblib pipeline
                raw_probs = pipeline.predict_proba(df)[:, 1]
                df['Risk_Score'] = raw_probs
                
                # Logic Flip for Bulk
                mask_yes = df['previous_loan_defaults_on_file'].str.lower() == 'yes'
                df['Risk_Score'] = np.where(mask_yes, 
                                            np.where(df['Risk_Score'] < 0.5, 1.0 - df['Risk_Score'], df['Risk_Score']),
                                            np.where(df['Risk_Score'] > 0.5, 1.0 - df['Risk_Score'], df['Risk_Score']))
                
                df['Status'] = np.where(df['Risk_Score'] < 0.25, '✅ APPROVED', '❌ REJECTED')
                
                st.success("Batch Analysis Complete!")
                st.plotly_chart(px.pie(df, names='Status', color='Status', color_discrete_map={'✅ APPROVED':'#10a37f','❌ REJECTED':'#ff4b4b'}))
                st.dataframe(df.style.applymap(lambda x: 'color: #10a37f; font-weight: bold' if x == '✅ APPROVED' else 'color: #ff4b4b; font-weight: bold', subset=['Status']))
                st.download_button("📥 Download Result CSV", df.to_csv(index=False).encode('utf-8'), "Analysis_Report.csv", "text/csv")
            except Exception as e:
                st.error(f"⚠️ Batch Prediction Error: {e}")

st.markdown("<div class='footer'>Developed @2026 by Prajwal Rajput</div>", unsafe_allow_html=True)
