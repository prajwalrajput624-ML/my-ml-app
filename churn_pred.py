import streamlit as st
import pandas as pd
import pickle
import time
import xgboost as xgb
from datetime import datetime
import io

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Churn Predictor Pro",
    page_icon="🚀",
    layout="wide"
)

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        with open("banker_churn.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

pipeline = load_model()

# ================= REQUIRED COLUMNS FOR VALIDATION =================
REQUIRED_COLUMNS = [
    "Gender", "Education_Level", "Marital_Status", "Income_Category", 
    "Card_Category", "Customer_Age", "Dependent_count", "Months_on_book", 
    "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", 
    "Credit_Limit", "Total_Revolving_Bal", "Total_Trans_Ct", 
    "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"
]

# ================= ADVANCED ANIMATED CSS =================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f2027);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        width: 100%; border-radius: 20px;
        background: linear-gradient(90deg, #ff8c00, #ff0080);
        color: white; font-weight: bold; border: none;
        transition: 0.3s all ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 20px rgba(255, 0, 128, 0.6);
    }
    .result-card {
        padding: 20px; border-radius: 15px;
        animation: slideIn 0.8s ease-out;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .churn-box { background: rgba(255, 75, 75, 0.15); border-left: 10px solid #ff4b4b; }
    .stay-box { background: rgba(0, 255, 153, 0.15); border-left: 10px solid #00ff99; }
    .pulse { animation: pulse-animation 2s infinite; }
    @keyframes pulse-animation {
        0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; }
    }
    .footer { text-align: center; padding: 20px; font-size: 12px; opacity: 0.6; }
</style>
""", unsafe_allow_html=True)

# ================= LOGIN PAGE =================
def login_page():
    st.markdown("<h1 style='text-align: center; color: white;'>🔐 Churn-AI Login</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            btn = st.form_submit_button("Launch Dashboard")
        if btn:
            if u == "prajwal" and p == "prajwal6575":
                st.session_state.logged_in = True
                st.balloons()
                st.rerun()
            else:
                st.error("Invalid Credentials")

# ================= MAIN APP =================
def main_app():
    if pipeline is None:
        st.error("Model file not found. Please upload 'banker_churn.pkl' to the directory.")
        return

    with st.sidebar:
        st.markdown("<h2 class='pulse'>⚙️ Control Center</h2>", unsafe_allow_html=True)
        threshold = st.slider("Model Sensitivity", 0.1, 0.9, 0.3, 0.05)
        page = st.selectbox("Navigate To", ["🔮 Predictor", "📂 Batch Process", "📜 Logs"])
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown(f"<h1>🚀 AI Customer Insights <small style='font-size:15px; color:#bbb;'>v4.0</small></h1>", unsafe_allow_html=True)

    if page == "🔮 Predictor":
        st.subheader("Customer Data Entry")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                Gender = st.selectbox("Gender", ["Male", "Female"])
                Education = st.selectbox("Education", ["Uneducated","High School","College","Graduate","Post-Graduate","Doctorate"])
                Income = st.selectbox("Annual Income", ["Less than $40K","$40K - $60K","$60K - $80K","$120K +"])
                Card = st.selectbox("Card Tier", ["Blue","Silver","Gold","Platinum"])
                Age = st.slider("Age", 18, 100, 35)
                Dependents = st.number_input("Dependents", 0, 10, 2)
            with col2:
                Tenure = st.number_input("Tenure (Months)", 1, 100, 24)
                Products = st.number_input("Products Used", 1, 10, 4)
                Inactive = st.number_input("Months Inactive", 0, 12, 2)
                Contacts = st.number_input("Bank Contacts", 0, 12, 2)
                Limit = st.number_input("Credit Limit", 500, 50000, 10000)
                Revolving = st.number_input("Unpaid Balance", 0, 50000, 1500)

        col3, col4 = st.columns(2)
        with col3: Trans_Ct = st.number_input("Total Transactions", 1, 300, 60)
        with col4: Trend = st.number_input("Activity Trend (Q4/Q1)", 0.0, 5.0, 1.2)
        Utilization = st.slider("Credit Utilization Rate", 0.0, 1.0, 0.3)

        if st.button("✨ Analyze Risk Now"):
            df = pd.DataFrame([[Gender, Education, "Single", Income, Card, Age, Dependents, Tenure, Products, Inactive, Contacts, Limit, Revolving, Trans_Ct, Trend, Utilization]], columns=REQUIRED_COLUMNS)
            with st.spinner("🤖 AI Thinking..."):
                time.sleep(1)
                raw_prob = pipeline.predict_proba(df)[:,1][0]
                prob_percent = round(float(raw_prob) * 100, 2)
                pred = int(raw_prob >= threshold)

            if pred:
                st.snow()
                st.markdown(f"<div class='result-card churn-box'><h2 style='color:#ff4b4b;'>⚠️ HIGH RISK DETECTED</h2><p>Probability: <b>{prob_percent}%</b></p></div>", unsafe_allow_html=True)
            else:
                st.balloons()
                st.markdown(f"<div class='result-card stay-box'><h2 style='color:#00ff99;'>✅ CUSTOMER IS LOYAL</h2><p>Probability: <b>{prob_percent}%</b></p></div>", unsafe_allow_html=True)
            st.session_state.history.insert(0, {"Time": datetime.now().strftime("%H:%M:%S"), "Status": "Risk" if pred else "Safe", "Score": f"{prob_percent}%"})

    elif page == "📂 Batch Process":
        st.subheader("CSV Intelligence Analysis")
        
        # Guide user to download template first
        template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("📋 Download Required CSV Template", template_csv, "template.csv", "text/csv")
        
        uploaded_file = st.file_uploader("Upload bank data CSV", type=["csv"])
        
        if uploaded_file:
            try:
                # Attempt to read the file
                data = pd.read_csv(uploaded_file)
                
                # VALIDATION: Check if columns match
                missing_cols = [col for col in REQUIRED_COLUMNS if col not in data.columns]
                
                if missing_cols:
                    st.error("❌ Invalid File Format!")
                    st.write(f"The uploaded file is missing the following required columns: **{', '.join(missing_cols)}**")
                    st.info("Please ensure your CSV matches the template provided above.")
                else:
                    with st.status("Processing Data...", expanded=True) as status:
                        st.write("Verifying data structure...")
                        time.sleep(1)
                        probs = pipeline.predict_proba(data[REQUIRED_COLUMNS])[:,1]
                        data["Risk_Score (%)"] = (probs * 100).round(2)
                        data["Prediction"] = ["LEAVING" if p >= threshold else "STAYING" for p in probs]
                        status.update(label="Analysis Complete!", state="complete", expanded=False)
                    
                    # Download Section
                    st.markdown("### 📥 Download Results")
                    csv_output = data.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Processed CSV", csv_output, f"churn_results_{datetime.now().strftime('%H%M')}.csv", "text/csv")
                    st.dataframe(data.style.background_gradient(subset=['Risk_Score (%)'], cmap='Reds'), use_container_width=True)

            except Exception as e:
                st.error("🚨 System Error: Could not process the file.")
                st.write("Make sure the file is a valid CSV and is not corrupted.")
                st.exception(e)

    elif page == "📜 Logs":
        st.subheader("Recent Activity Logs")
        if st.session_state.history:
            st.table(pd.DataFrame(st.session_state.history))
        else: st.info("No activity recorded yet.")

    st.markdown(f"<div class='footer'>AI Core: XGBoost | Developed by Prajwal Rajput | {datetime.now().year}</div>", unsafe_allow_html=True)

if st.session_state.logged_in: main_app()
else: login_page()

