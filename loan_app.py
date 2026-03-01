import streamlit as st
import pandas as pd
import pickle
import time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Loan Approval AI System", layout="wide")

# ================= STYLE =================
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
.stButton>button { width: 100%; background: #10a37f; color: white; border-radius: 8px; height: 3em; }
.stButton>button:hover { background: #0d8a6a; }
.header { text-align:center; color:#10a37f; }
.footer { position: fixed; bottom: 0; width: 100%; background: #161b22;
          text-align:center; padding:10px; color:#8b949e; }
</style>
""", unsafe_allow_html=True)

# ================= LOGIN SYSTEM =================
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown("<h1 class='header'>🔐 Secure Login</h1>", unsafe_allow_html=True)
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if u == "prajwal" and p == "prajwal6575":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Invalid Credentials")
    st.stop()

# ================= MODEL LOADING =================
@st.cache_resource
def load_model():
    model_path = "loan_model.pkl"

    if not os.path.exists(model_path):
        st.error("❌ loan_model.pkl not found.")
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Safety check
        if isinstance(model, str):
            st.error("❌ Model file contains string, not trained model.")
            return None

        if not hasattr(model, "predict_proba"):
            st.error("❌ Loaded object is not a valid sklearn model.")
            return None

        return model

    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None


pipeline = load_model()
if pipeline is None:
    st.stop()

# ================= REQUIRED FEATURES =================
required_cols = [
    'person_age', 'person_income', 'person_emp_exp',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score',
    'person_gender', 'person_education',
    'person_home_ownership', 'loan_intent',
    'previous_loan_defaults_on_file'
]

# ================= SIDEBAR =================
st.sidebar.write("Active Session: Prajwal Rajput")
mode = st.sidebar.radio("Navigation", ["Individual Scan", "Batch CSV Scan"])

if st.sidebar.button("Logout"):
    st.session_state.auth = False
    st.rerun()

# =====================================================
# ================= INDIVIDUAL MODE ===================
# =====================================================
if mode == "Individual Scan":

    st.markdown("<h2 class='header'>Single Applicant Analysis</h2>", unsafe_allow_html=True)

    with st.form("single"):
        col1, col2 = st.columns(2)

        with col1:
            income = st.number_input("Annual Income", min_value=0.0, value=45000.0)
            loan = st.number_input("Loan Amount", min_value=0.0, value=12000.0)
            score = st.number_input("Credit Score", 300, 850, 700)
            rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)

        with col2:
            default = st.selectbox("Previous Default", ["no", "yes"])
            home = st.selectbox("Home Ownership", ["rent", "own", "mortgage"])
            intent = st.selectbox("Loan Purpose", ["personal", "education", "medical", "venture"])
            exp = st.number_input("Job Experience (Years)", min_value=0.0, value=5.0)

        submit = st.form_submit_button("Run AI Scan")

    if submit:

        loan_percent = loan / income if income != 0 else 0

        data = pd.DataFrame([{
            'person_age': 25,
            'person_income': income,
            'person_emp_exp': exp,
            'loan_amnt': loan,
            'loan_int_rate': rate,
            'loan_percent_income': loan_percent,
            'cb_person_cred_hist_length': 5.0,
            'credit_score': score,
            'person_gender': 'male',
            'person_education': 'bachelor',
            'person_home_ownership': home,
            'loan_intent': intent,
            'previous_loan_defaults_on_file': default
        }])

        data = data[required_cols]

        try:
            prob = pipeline.predict_proba(data)[0][1]
            risk = prob

            c1, c2 = st.columns(2)

            with c1:
                if risk < 0.25:
                    st.success("✅ APPROVED")
                    st.metric("Confidence", f"{100-(risk*100):.2f}%")
                    st.balloons()
                else:
                    st.error("❌ REJECTED")
                    st.metric("Risk Score", f"{risk:.2%}")

            with c2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk*100,
                    gauge={'axis': {'range': [0, 100]}},
                    title={'text': "Risk Level"}
                ))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =====================================================
# ================= BATCH MODE ========================
# =====================================================
else:

    st.markdown("<h2 class='header'>Batch Loan Processing</h2>", unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV File", type="csv")

    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing Columns: {missing}")
            st.stop()

        st.write(f"Rows Loaded: {len(df)}")

        if st.button("Run Batch Analysis"):

            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            df = df[required_cols]

            try:
                probs = pipeline.predict_proba(df)[:, 1]
                df["Risk_Score"] = probs
                df["Status"] = np.where(df["Risk_Score"] < 0.25,
                                        "APPROVED",
                                        "REJECTED")

                st.success("Batch Analysis Completed")

                col1, col2 = st.columns(2)

                with col1:
                    counts = df["Status"].value_counts().reset_index()
                    counts.columns = ["Status", "Count"]
                    fig = px.bar(counts, x="Status", y="Count",
                                 template="plotly_dark",
                                 text="Count")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results CSV",
                                       data=csv,
                                       file_name="loan_results.csv",
                                       mime="text/csv")

                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Batch Prediction Error: {e}")

st.markdown("<div class='footer'>Developed @2026 by Prajwal Rajput</div>", unsafe_allow_html=True)
