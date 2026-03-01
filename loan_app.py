import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 1. CONFIG =================
pd.set_option("styler.render.max_elements", 1500000)
st.set_page_config(page_title="Loan Approval AI-System", layout="wide")

# ================= 2. STYLING =================
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stButton>button { width: 100%; border-radius: 10px; background: #10a37f; color: white; font-weight: bold; height: 3em; }
    .main-header { text-align: center; color: #10a37f; margin-bottom: 20px; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #161b22; color: #8b949e; text-align: center;
        padding: 10px; border-top: 1px solid #30363d; z-index: 100; }
    </style>
""", unsafe_allow_html=True)

# ================= 3. LOGIN =================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h1 class='main-header'>🛡️ Login</h1>", unsafe_allow_html=True)
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

# ================= 4. MODEL LOAD =================
try:
    pipeline = joblib.load("loan_model.joblib")
except FileNotFoundError:
    st.error("loan_model.joblib file not found.")
    st.stop()

# ================= 5. SIDEBAR =================
st.sidebar.success("Logged in: Prajwal Rajput")
mode = st.sidebar.radio("Navigation", ["Individual Scan", "Upload CSV"])

if st.sidebar.button("🔒 Logout"):
    st.session_state.authenticated = False
    st.rerun()

# =====================================================
# =============== INDIVIDUAL MODE =====================
# =====================================================

if mode == "Individual Scan":

    st.markdown("<h2 class='main-header'>Individual Profile Analysis</h2>", unsafe_allow_html=True)

    with st.form("single_entry"):
        c1, c2 = st.columns(2)

        with c1:
            income = st.number_input("Annual Income ($)", min_value=1, value=45000)
            loan = st.number_input("Loan Amount ($)", min_value=0, value=12000)
            score = st.number_input("Credit Score", 300, 850, 710)
            rate = st.number_input("Interest Rate (%)", value=10.0)

        with c2:
            default_text = st.selectbox("Previous Default", ["no", "yes"])
            home = st.selectbox("Home Ownership", ["rent", "own", "mortgage"])
            intent = st.selectbox("Loan Purpose", ["personal", "education", "medical", "venture"])
            exp = st.number_input("Job Experience (Years)", min_value=0, value=5)

        btn = st.form_submit_button("🚀 Run AI Analysis")

    if btn:

        with st.spinner("AI analyzing profile..."):
            time.sleep(1)

            # FIX 1 → yes/no to 1/0
            default = 1 if default_text == "yes" else 0

            # FIX 2 → zero division safe
            percent_income = loan / income if income != 0 else 0

            # Create DataFrame EXACT matching training
            full_data = pd.DataFrame([{
                'person_age': 25,
                'person_income': float(income),
                'person_emp_exp': float(exp),
                'loan_amnt': float(loan),
                'loan_int_rate': float(rate),
                'loan_percent_income': float(percent_income),
                'cb_person_cred_hist_length': 5,
                'credit_score': float(score),
                'person_gender': 'male',
                'person_education': 'bachelor',
                'person_home_ownership': home,
                'loan_intent': intent,
                'previous_loan_defaults_on_file': default
            }])

            try:
                raw_prob = pipeline.predict_proba(full_data)[0][1]
            except Exception as e:
                st.error(f"Model Prediction Error: {e}")
                st.write("Columns expected:", pipeline.feature_names_in_)
                st.write("Columns given:", full_data.columns)
                st.stop()

            risk = raw_prob

        st.divider()
        col1, col2 = st.columns([1.5,1])

        with col1:
            if risk < 0.25:
                st.success(f"### ✅ APPROVED\nConfidence: {(1-risk)*100:.2f}%")
                st.balloons()
            else:
                st.error(f"### ❌ REJECTED\nRisk Score: {risk:.2%}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk*100,
                gauge={'axis': {'range': [0, 100]}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ================= BATCH MODE ========================
# =====================================================

else:

    st.markdown("<h2 class='main-header'>Batch Processing</h2>", unsafe_allow_html=True)

    file = st.file_uploader("Upload Applicant CSV", type="csv")

    if file:
        df = pd.read_csv(file)

        # Required columns check
        required_cols = list(pipeline.feature_names_in_)
        if not all(col in df.columns for col in required_cols):
            st.error("CSV columns do not match model training columns.")
            st.write("Required Columns:", required_cols)
            st.stop()

        if st.button("⚡ Execute AI Scan"):

            with st.spinner("Processing data..."):
                raw_probs = pipeline.predict_proba(df)[:, 1]
                df["Risk_Score"] = raw_probs
                df["Status"] = np.where(df["Risk_Score"] < 0.25,
                                        "✅ APPROVED",
                                        "❌ REJECTED")

            st.success("Batch Analysis Complete!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df))
            col2.metric("Approved", len(df[df["Status"]=="✅ APPROVED"]))
            col3.metric("Rejected", len(df[df["Status"]=="❌ REJECTED"]))

            fig = px.pie(df, names="Status")
            st.plotly_chart(fig)

            st.dataframe(df.head(1000))

            st.download_button(
                "Download Full Report",
                df.to_csv(index=False).encode("utf-8"),
                "Loan_Report.csv",
                "text/csv"
            )

# ================= FOOTER =================
st.markdown("<div class='footer'>Developed @2026 by Prajwal Rajput</div>", unsafe_allow_html=True)
