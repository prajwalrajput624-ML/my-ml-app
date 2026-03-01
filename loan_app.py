import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ================= 1. SYSTEM CONFIG =================
pd.set_option("styler.render.max_elements", 1500000)
st.set_page_config(page_title="FinGuard AI | Prajwal Rajput", layout="wide")

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
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Launch System"):
                if u == "prajwal" and p == "prajwal6575":
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("Invalid Credentials")
    st.stop()

# ================= 3. MODEL LOAD =================
@st.cache_resource
def load_my_model():
    try:
        return joblib.load('loan_model.joblib')
    except:
        return None

pipeline = load_my_model()
if pipeline is None:
    st.error("❌ 'loan_model.joblib' not found! Please upload it to the same folder.")
    st.stop()

# ================= 4. SIDEBAR =================
st.sidebar.write(f"User: **Prajwal Rajput**")
mode = st.sidebar.radio("Navigation", ["Individual Scan", "Bulk CSV Process"])
if st.sidebar.button("🔒 Logout"):
    st.session_state.authenticated = False
    st.rerun()

# ================= 5. CORE PREDICTION LOGIC =================
def process_prediction(input_df):
    try:
        # EXACT COLUMN ORDER FROM YOUR CSV
        columns = [
            'person_age', 'person_gender', 'person_education', 'person_income',
            'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'previous_loan_defaults_on_file'
        ]
        
        # Re-ordering and data type fix
        input_df = input_df[columns].copy()
        
        # Numeric conversion
        num_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
        for col in num_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Prediction
        raw_probs = pipeline.predict_proba(input_df)[:, 1]
        
        # Logic Flip based on default status
        final_risks = []
        for i, prob in enumerate(raw_probs):
            default_val = str(input_df.iloc[i]['previous_loan_defaults_on_file']).lower()
            if default_val == 'yes':
                risk = 1.0 - prob if prob < 0.5 else prob
            else:
                risk = prob if prob < 0.5 else 1.0 - prob
            final_risks.append(risk)
            
        return np.array(final_risks)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# ================= 6. MODE 1: INDIVIDUAL SCAN =================
if mode == "Individual Scan":
    st.markdown("<h2 class='main-header'>Single Applicant Analysis</h2>", unsafe_allow_html=True)
    with st.form("single"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 100, 25)
            gender = st.selectbox("Gender", ["male", "female"])
            edu = st.selectbox("Education", ["Bachelor", "Master", "High School", "Associate"])
            income = st.number_input("Income ($)", value=50000)
            exp = st.number_input("Experience (Yrs)", value=5)
            home = st.selectbox("Home", ["RENT", "OWN", "MORTGAGE"])
        with c2:
            loan = st.number_input("Loan Amount ($)", value=10000)
            intent = st.selectbox("Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            rate = st.number_input("Interest Rate (%)", value=11.0)
            score = st.number_input("Credit Score", 300, 850, 700)
            hist = st.number_input("Credit History Length", value=5)
            default = st.selectbox("Past Defaults?", ["No", "Yes"])
        
        if st.form_submit_button("🚀 START AI SCAN"):
            test_df = pd.DataFrame([{
                'person_age': age, 'person_gender': gender, 'person_education': edu,
                'person_income': income, 'person_emp_exp': exp, 'person_home_ownership': home,
                'loan_amnt': loan, 'loan_intent': intent, 'loan_int_rate': rate,
                'loan_percent_income': loan/income, 'cb_person_cred_hist_length': hist,
                'credit_score': score, 'previous_loan_defaults_on_file': default
            }])
            
            risk_res = process_prediction(test_df)
            if risk_res is not None:
                risk = risk_res[0]
                st.divider()
                col_res1, col_res2 = st.columns([1.5, 1])
                with col_res1:
                    if risk < 0.25:
                        st.success(f"### ✅ APPROVED\n**Safety Score:** {100-risk*100:.2f}%")
                        st.balloons()
                    else:
                        st.error(f"### ❌ REJECTED\n**Risk Level:** {risk:.2%}")
                with col_res2:
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=risk*100, 
                                               gauge={'bar': {'color': "#10a37f" if risk < 0.25 else "#ff4b4b"}}))
                    fig.update_layout(height=250, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color="white")
                    st.plotly_chart(fig, use_container_width=True)

# ================= 7. MODE 2: BATCH PROCESS =================
else:
    st.markdown("<h2 class='main-header'>Bulk Applicant Processing</h2>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        if st.button("⚡ PROCESS ALL RECORDS"):
            results = process_prediction(df)
            if results is not None:
                df['Risk_Score'] = results
                df['Status'] = np.where(df['Risk_Score'] < 0.25, '✅ APPROVED', '❌ REJECTED')
                
                st.success(f"Processed {len(df)} records!")
                st.plotly_chart(px.pie(df, names='Status', color='Status', 
                                     color_discrete_map={'✅ APPROVED':'#10a37f','❌ REJECTED':'#ff4b4b'}))
                
                st.dataframe(df.style.applymap(lambda x: 'color: #10a37f; font-weight: bold' if x == '✅ APPROVED' else 'color: #ff4b4b; font-weight: bold', subset=['Status']))
                st.download_button("📥 Download Results", df.to_csv(index=False).encode('utf-8'), "Prajwal_Batch_Report.csv", "text/csv")

st.markdown("<div class='footer'>Developed by Prajwal Rajput | AI Financial Intelligence</div>", unsafe_allow_html=True)
