import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 1. SYSTEM CONFIG (FIXES YOUR ERROR) =================
# Styling limit ko 1.5 Million cells tak badha diya hai
pd.set_option("styler.render.max_elements", 1500000)

st.set_page_config(page_title="Loan Approval AI-System", layout="wide")

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
    .card { background-color: #1c2128; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. LOGIN SYSTEM =================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h1 class='main-header'>🛡️Login</h1>", unsafe_allow_html=True)
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
    st.markdown("<div class='footer'>Developed by Prajwal Rajput</div>", unsafe_allow_html=True)
    st.stop()

# ================= 3. MODEL LOAD =================
try:
    pipeline = joblib.load('loan_model.joblib')
except:
    st.error("Error: 'loan_model.joblib' file not found.")
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
            income = st.number_input("Annual Income ($)", value=45000)
            loan = st.number_input("Loan Amount ($)", value=12000)
            score = st.number_input("Credit Score", 300, 850, 710)
            rate = st.number_input("Interest Rate (%)", value=10.0)
        with c2:
            default = st.selectbox("Previous Default", ["no", "yes"])
            home = st.selectbox("Home Ownership", ["rent", "own", "mortgage"])
            intent = st.selectbox("Loan Purpose", ["personal", "education", "medical", "venture"])
            exp = st.number_input("Job Experience (Years)", value=5)
        btn = st.form_submit_button("🚀 Running Loan-Approval AI")

    if btn:
        with st.spinner("AI is analyzing..."):
            time.sleep(1)
            full_data = pd.DataFrame([{
                'person_age': 25, 'person_income': income, 'person_emp_exp': exp,
                'loan_amnt': loan, 'loan_int_rate': rate, 'loan_percent_income': loan/income,
                'cb_person_cred_hist_length': 5, 'credit_score': score,
                'person_gender': 'male', 'person_education': 'bachelor',
                'person_home_ownership': home, 'loan_intent': intent,
                'previous_loan_defaults_on_file': default
            }])
            raw_prob = pipeline.predict_proba(full_data)[0][1]
            risk = (1.0 - raw_prob if raw_prob < 0.5 else raw_prob) if default == 'yes' else (raw_prob if raw_prob < 0.5 else 1.0 - raw_prob)
            
            st.divider()
            res_c1, res_c2 = st.columns([1.5, 1])
            with res_c1:
                if risk < 0.25:
                    st.success(f"### ✅ APPROVED | Confidence: {100-(risk*100):.2f}%")
                    st.info(f"**AI Insight:** Reliable profile based on credit history.")
                    st.balloons()
                else:
                    st.error(f"### ❌ REJECTED | Risk: {risk:.2%}")
                    st.warning(f"**Primary Risk:** Previous defaults or low score detected.")
            with res_c2:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=risk*100, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#10a37f"}}))
                fig.update_layout(height=230, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

# ================= 6. MODE 2: HIGH-CAPACITY BATCH SCAN =================
else:
    st.markdown("<h2 class='main-header'>High-Capacity Batch Processing</h2>", unsafe_allow_html=True)
    file = st.file_uploader("Upload Applicant CSV", type="csv")
    
    if file:
        df = pd.read_csv(file)
        num_cells = df.shape[0] * df.shape[1]
        st.write(f"📊 Records: {len(df)} | Total Cells: {num_cells}")
        
        if st.button("⚡ EXECUTE MASSIVE SCAN"):
            with st.spinner("AI Engine is crunching massive data..."):
                # Vectorized Processing
                raw_probs = pipeline.predict_proba(df)[:, 1]
                df['Risk_Score'] = raw_probs
                mask_yes = df['previous_loan_defaults_on_file'].str.lower() == 'yes'
                mask_no = df['previous_loan_defaults_on_file'].str.lower() == 'no'
                
                df.loc[mask_yes, 'Risk_Score'] = np.where(df.loc[mask_yes, 'Risk_Score'] < 0.5, 1.0 - df.loc[mask_yes, 'Risk_Score'], df.loc[mask_yes, 'Risk_Score'])
                df.loc[mask_no, 'Risk_Score'] = np.where(df.loc[mask_no, 'Risk_Score'] > 0.5, 1.0 - df.loc[mask_no, 'Risk_Score'], df.loc[mask_no, 'Risk_Score'])
                
                df['Status'] = np.where(df['Risk_Score'] < 0.25, '✅ APPROVED', '❌ REJECTED')

                # --- DASHBOARD STATS ---
                st.success("Analysis Complete!")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Rows", len(df))
                col_b.metric("Approvals", len(df[df['Status'] == '✅ APPROVED']))
                col_c.metric("Rejections", len(df[df['Status'] == '❌ REJECTED']))

                # Pie Chart
                fig_pie = px.pie(df, names='Status', color='Status', color_discrete_map={'✅ APPROVED':'#10a37f','❌ REJECTED':'#ff4b4b'})
                st.plotly_chart(fig_pie)

                # --- LARGE DATA RENDERING FIX ---
                st.write("### Data Report View")
                if num_cells < 1500000:
                    styled_df = df.style.applymap(lambda x: 'color: #10a37f; font-weight: bold' if x == '✅ APPROVED' else 'color: #ff4b4b; font-weight: bold', subset=['Status'])
                    st.dataframe(styled_df)
                else:
                    st.warning("⚠️ Data is extremely large. Rendering top 1000 rows with styling. Download full CSV for all results.")
                    st.dataframe(df.head(1000).style.applymap(lambda x: 'color: #10a37f; font-weight: bold' if x == '✅ APPROVED' else 'color: #ff4b4b; font-weight: bold', subset=['Status']))

                st.download_button("📥 Download Full Result CSV", df.to_csv(index=False).encode('utf-8'), "Report.csv", "text/csv")

# ================= 7. FOOTER =================
st.markdown("<div class='footer'>Developed @2026 by Prajwal Rajput</div>", unsafe_allow_html=True)
