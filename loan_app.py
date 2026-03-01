import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= CONFIG =================
pd.set_option("styler.render.max_elements", 1500000)
st.set_page_config(page_title="Loan Approval AI-System", layout="wide")

# ================= STYLE =================
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
.stButton>button { width: 100%; border-radius: 10px; background: #10a37f; color: white; height: 3em; font-weight: bold; }
.main-header { text-align: center; color: #10a37f; }
.footer { position: fixed; left: 0; bottom: 0; width: 100%;
background-color: #161b22; color: #8b949e; text-align: center;
padding: 10px; border-top: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h1 class='main-header'>🛡️ Login</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1.5,1])
    with col2:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if u == "prajwal" and p == "prajwal6575":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
    st.stop()

# ================= LOAD MODEL =================
try:
    pipeline = joblib.load("loan_model.joblib")
except FileNotFoundError:
    st.error("loan_model.joblib not found.")
    st.stop()

# ================= SIDEBAR =================
st.sidebar.success("Logged in: Prajwal Rajput")
mode = st.sidebar.radio("Navigation", ["Individual Scan", "Upload CSV"])

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# ==================================================
# ================ INDIVIDUAL MODE =================
# ==================================================

if mode == "Individual Scan":

    st.markdown("<h2 class='main-header'>Individual Profile Analysis</h2>", unsafe_allow_html=True)

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            income = st.number_input("Annual Income ($)", min_value=1, value=45000)
            loan = st.number_input("Loan Amount ($)", min_value=0, value=12000)
            score = st.number_input("Credit Score", 300, 850, 710)
            rate = st.number_input("Interest Rate (%)", value=10.0)

        with col2:
            default = st.selectbox("Previous Default", ["no", "yes"])
            home = st.selectbox("Home Ownership", ["rent", "own", "mortgage"])
            intent = st.selectbox("Loan Purpose", ["personal", "education", "medical", "venture"])
            exp = st.number_input("Job Experience (Years)", min_value=0, value=5)

        submit = st.form_submit_button("🚀 Run AI")

    if submit:

        percent_income = loan / income if income != 0 else 0

        # EXACT TRAINING ORDER
        full_data = pd.DataFrame([[
            25,                              # person_age
            float(income),
            float(exp),
            float(loan),
            float(rate),
            float(percent_income),
            5,                               # cb_person_cred_hist_length
            float(score),
            "male",                          # person_gender
            "bachelor",                      # person_education
            home,
            intent,
            default                           # string yes/no
        ]], columns=[
            'person_age',
            'person_income',
            'person_emp_exp',
            'loan_amnt',
            'loan_int_rate',
            'loan_percent_income',
            'cb_person_cred_hist_length',
            'credit_score',
            'person_gender',
            'person_education',
            'person_home_ownership',
            'loan_intent',
            'previous_loan_defaults_on_file'
        ])

        try:
            raw_prob = pipeline.predict_proba(full_data)[0][1]
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.write("Expected Columns:", pipeline.feature_names_in_)
            st.write("Given Columns:", full_data.columns.tolist())
            st.stop()

        risk = raw_prob

        st.divider()
        colA, colB = st.columns([1.5,1])

        with colA:
            if risk < 0.25:
                st.success(f"### ✅ APPROVED\nConfidence: {(1-risk)*100:.2f}%")
                st.balloons()
            else:
                st.error(f"### ❌ REJECTED\nRisk Score: {risk:.2%}")

        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk*100,
                gauge={'axis': {'range': [0, 100]}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

# ==================================================
# ================= BATCH MODE =====================
# ==================================================

else:

    st.markdown("<h2 class='main-header'>Batch Processing</h2>", unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)

        required_cols = list(pipeline.feature_names_in_)

        if not all(col in df.columns for col in required_cols):
            st.error("CSV columns mismatch with model training.")
            st.write("Required Columns:", required_cols)
            st.stop()

        if st.button("⚡ Run Batch AI"):

            raw_probs = pipeline.predict_proba(df)[:, 1]
            df["Risk_Score"] = raw_probs
            df["Status"] = np.where(df["Risk_Score"] < 0.25,
                                    "✅ APPROVED",
                                    "❌ REJECTED")

            st.success("Batch Analysis Complete")

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
