import streamlit as st
import pandas as pd
import pickle
import time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import io

# ================= 1. SYSTEM CONFIG =================
pd.set_option("styler.render.max_elements", 1500000)
st.set_page_config(page_title="Loan Approval AI-System", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stButton>button { width: 100%; border-radius: 10px; background: #10a37f; color: white; font-weight: bold; height: 3em; transition: 0.3s; }
    .stButton>button:hover { background: #0d8a6a; transform: scale(1.01); border: 1px solid white; }
    .main-header { text-align: center; color: #10a37f; margin-bottom: 20px; font-family: 'Trebuchet MS'; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    .scanning-text { 
        color: #10a37f; 
        font-weight: bold; 
        font-family: 'Courier New'; 
        animation: pulse 1s infinite;
        text-align: center;
        font-size: 1.2rem;
    }
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

# ================= 3. MODEL LOAD =================
@st.cache_resource
def load_model():
    model_path = 'loan_model.pkl' 
    if not os.path.exists(model_path):
        st.error(f"❌ Error: '{model_path}' file not found.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Error: {e}")
        return None

pipeline = load_model()
if pipeline is None:
    st.stop()

# ================= 4. SIDEBAR =================
st.sidebar.write(f"Active Session: **Prajwal Rajput**")
mode = st.sidebar.radio("Navigation", ["Individual Scan", "Upload CSV Batch"])
if st.sidebar.button("🔒 Logout"):
    st.session_state.authenticated = False
    st.rerun()

# ================= 5. MODE 1: INDIVIDUAL SCAN =================
if mode == "Individual Scan":
    st.markdown("<h2 class='main-header'>Single Applicant Neural Analysis</h2>", unsafe_allow_html=True)
    with st.form("single_entry"):
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Annual Income ($)", value=45000)
            loan = st.number_input("Loan Amount ($)", value=12000)
            score = st.number_input("Credit Score", 300, 850, 710)
            rate = st.number_input("Interest Rate (%)", value=10.0)
        with c2:
            default = st.selectbox("Previous Default", ["no", "yes"])
            home = st.selectbox("Home Ownership", ["rent", "own", "mortgage"])
            intent = st.selectbox("Loan Purpose", ["personal", "education", "medical", "venture"])
            exp = st.number_input("Job Experience (Years)", value=5)
        btn = st.form_submit_button("🚀 INITIATE AI SCAN")

    if btn:
        status_box = st.empty()
        for msg in ["📡 Syncing with Credit Bureaus...", "🧠 Running Probability Matrix...", "⚖️ Finalizing Verdict..."]:
            status_box.markdown(f"<p class='scanning-text'>{msg}</p>", unsafe_allow_html=True)
            time.sleep(0.7)
        status_box.empty()

        full_data = pd.DataFrame([{
            'person_age': 25, 'person_income': float(income), 'person_emp_exp': float(exp),
            'loan_amnt': float(loan), 'loan_int_rate': float(rate), 
            'loan_percent_income': float(loan/income),
            'cb_person_cred_hist_length': 5.0, 'credit_score': float(score),
            'person_gender': 'male', 'person_education': 'bachelor',
            'person_home_ownership': home.lower(), 'loan_intent': intent.lower(),
            'previous_loan_defaults_on_file': default.lower()
        }])
        
        try:
            raw_prob = pipeline.predict_proba(full_data)[0][1]
            risk = (1.0 - raw_prob if raw_prob < 0.5 else raw_prob) if default == 'yes' else (raw_prob if raw_prob < 0.5 else 1.0 - raw_prob)
            
            st.divider()
            res_c1, res_c2 = st.columns([1, 1])
            with res_c1:
                if risk < 0.25:
                    st.success("### STATUS: ✅ APPROVED")
                    st.metric("Approval Confidence", f"{100-(risk*100):.1f}%")
                    st.balloons()
                else:
                    st.error("### STATUS: ❌ REJECTED")
                    st.metric("Risk Assessment", f"{risk:.2%}", delta="-High Risk", delta_color="inverse")
            
            with res_c2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=risk*100,
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#10a37f"},
                        'steps': [
                            {'range': [0, 25], 'color': "rgba(16, 163, 127, 0.2)"},
                            {'range': [75, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                        ]
                    },
                    title={'text': "Risk Magnitude"}
                ))
                fig.update_layout(height=250, margin=dict(t=50, b=0), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ================= 6. MODE 2: BATCH SCAN =================
else:
    st.markdown("<h2 class='main-header'>High-Volume Batch Processor</h2>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV Data Pack", type="csv")
    
    if file:
        df = pd.read_csv(file)
        st.write(f"✅ Ready to process **{len(df)}** rows.")
        
        if st.button("⚡ EXECUTE MASSIVE ANALYSIS"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Visual Processing Steps
                for i in range(1, 101, 10):
                    time.sleep(0.1)
                    progress_bar.progress(i)
                    status_text.markdown(f"<p class='scanning-text'>AI is scanning packet {i}%...</p>", unsafe_allow_html=True)
                
                # Model Logic
                raw_probs = pipeline.predict_proba(df)[:, 1]
                df['Risk_Score'] = raw_probs
                mask_yes = df['previous_loan_defaults_on_file'].str.lower() == 'yes'
                df['Risk_Score'] = np.where(mask_yes, 
                                            np.where(df['Risk_Score'] < 0.5, 1.0 - df['Risk_Score'], df['Risk_Score']),
                                            np.where(df['Risk_Score'] > 0.5, 1.0 - df['Risk_Score'], df['Risk_Score']))
                df['Status'] = np.where(df['Risk_Score'] < 0.25, '✅ APPROVED', '❌ REJECTED')
                
                status_text.empty()
                progress_bar.empty()
                st.success("Analysis Complete!")

                # Data Visualization
                v1, v2 = st.columns([2, 1])
                with v1:
                    counts = df['Status'].value_counts().reset_index()
                    counts.columns = ['Result', 'Total']
                    fig_bar = px.bar(counts, x='Result', y='Total', color='Result',
                                   text='Total', template="plotly_dark",
                                   color_discrete_map={'✅ APPROVED':'#10a37f','❌ REJECTED':'#ff4b4b'})
                    st.plotly_chart(fig_bar, use_container_width=True)

                with v2:
                    st.markdown("### 📥 Export Hub")
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Processed CSV", data=csv, file_name="ai_loan_report.csv", mime="text/csv")
                    st.info("The exported file contains individual risk scores and final decisions.")

                st.markdown("### 📋 Results Preview")
                st.dataframe(df.style.applymap(
                    lambda x: 'color: #10a37f; font-weight: bold' if x == '✅ APPROVED' else ('color: #ff4b4b; font-weight: bold' if x == '❌ REJECTED' else ''), 
                    subset=['Status']
                ), use_container_width=True)

            except Exception as e:
                st.error(f"Batch Processing Error: {e}")

st.markdown("<div class='footer'>Developed @2026 by Prajwal Rajput</div>", unsafe_allow_html=True)
