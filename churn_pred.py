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
    page_icon="🤖",
    layout="wide"
)

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# ================= LOAD MODEL & VALIDATION =================
@st.cache_resource
def load_model():
    try:
        with open("banker_churn.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

pipeline = load_model()

# Model ke liye zaroori columns
REQUIRED_COLUMNS = [
    "Gender", "Education_Level", "Marital_Status", "Income_Category", 
    "Card_Category", "Customer_Age", "Dependent_count", "Months_on_book", 
    "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", 
    "Credit_Limit", "Total_Revolving_Bal", "Total_Trans_Ct", 
    "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"
]

# ================= CUSTOM CSS (Animations & Glassmorphism) =================
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
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        width: 100%; border-radius: 25px;
        background: linear-gradient(90deg, #00dbde, #fc00ff);
        color: white; font-weight: bold; border: none;
        transition: 0.4s all ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 20px rgba(0, 219, 222, 0.6);
    }
    .result-card {
        padding: 30px; border-radius: 20px; text-align: center;
        animation: zoomIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1);
    }
    @keyframes zoomIn {
        from { opacity: 0; transform: scale(0.5); }
        to { opacity: 1; transform: scale(1); }
    }
    .churn-box { background: rgba(255, 75, 75, 0.2); border: 2px solid #ff4b4b; }
    .stay-box { background: rgba(0, 255, 153, 0.2); border: 2px solid #00ff99; }
    .pulse { animation: pulse-animation 2s infinite; }
    @keyframes pulse-animation {
        0% { text-shadow: 0 0 5px #fff; }
        50% { text-shadow: 0 0 20px #00dbde; }
        100% { text-shadow: 0 0 5px #fff; }
    }
</style>
""", unsafe_allow_html=True)

# ================= LOGIN PAGE =================
def login_page():
    st.markdown("<br><br><h1 style='text-align: center;' class='pulse'>🛰️ AI COMMAND CENTER</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1.5,1])
    with col2:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("INITIALIZE"):
                if u == "prajwal" and p == "prajwal6575":
                    st.session_state.logged_in = True
                    st.balloons()
                    st.rerun()
                else: st.error("Access Denied")

# ================= MAIN APP =================
def main_app():
    if pipeline is None:
        st.error("Model file 'banker_churn.pkl' missing! System halted.")
        return

    with st.sidebar:
        st.markdown("<h2 class='pulse'>⚙️ Engine Settings</h2>", unsafe_allow_html=True)
        threshold = st.slider("Risk Threshold", 0.1, 0.9, 0.3, 0.05)
        page = st.selectbox("Navigation", ["🔮 Risk Scanner", "📂 Batch Analysis", "📜 History"])
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown(f"<h1>🚀 Neural Insights <small style='font-size:15px; color:#00dbde;'>v4.0</small></h1>", unsafe_allow_html=True)

    # PAGE 1: SINGLE PREDICTION
    if page == "🔮 Risk Scanner":
        st.subheader("Manual Data Entry")
        c1, c2 = st.columns(2)
        with c1:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Edu = st.selectbox("Education", ["Uneducated","High School","College","Graduate","Post-Graduate","Doctorate"])
            Income = st.selectbox("Income Range", ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +"])
            Card = st.selectbox("Card Tier", ["Blue","Silver","Gold","Platinum"])
            Age = st.slider("Age", 18, 100, 35)
            Deps = st.number_input("Dependents", 0, 10, 2)
        with c2:
            Tenure = st.number_input("Tenure (Months)", 1, 100, 24)
            Products = st.number_input("Total Products", 1, 10, 4)
            Inact = st.number_input("Months Inactive", 0, 12, 2)
            Cont = st.number_input("Bank Contacts", 0, 12, 2)
            Limit = st.number_input("Credit Limit", 500, 50000, 10000)
            Bal = st.number_input("Unpaid Balance", 0, 50000, 1500)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1: Trans_Ct = st.number_input("Total Transactions", 1, 300, 60)
        with col_t2: Trend = st.number_input("Usage Trend (Q4/Q1)", 0.0, 5.0, 1.2)
        Util = st.slider("Utilization Rate", 0.0, 1.0, 0.3)

        if st.button("🔥 START SCAN"):
            df = pd.DataFrame([[Gender, Edu, "Single", Income, Card, Age, Deps, Tenure, Products, Inact, Cont, Limit, Bal, Trans_Ct, Trend, Util]], columns=REQUIRED_COLUMNS)
            with st.spinner("🤖 Processing..."):
                time.sleep(1)
                prob = pipeline.predict_proba(df)[:,1][0]
                prob_pct = round(float(prob) * 100, 2)
                is_churn = prob >= threshold

            if is_churn:
                st.snow()
                st.markdown(f"<div class='result-card churn-box'><h1 style='color:#ff4b4b;'>🚨 RISK: {prob_pct}%</h1><p>Customer likely to LEAVE</p></div>", unsafe_allow_html=True)
            else:
                st.balloons()
                st.markdown(f"<div class='result-card stay-box'><h1 style='color:#00ff99;'>✅ SAFE: {prob_pct}%</h1><p>Customer likely to STAY</p></div>", unsafe_allow_html=True)
            st.session_state.history.insert(0, {"Time": datetime.now().strftime("%H:%M"), "Score": f"{prob_pct}%", "Result": "Risk" if is_churn else "Safe"})

    # PAGE 2: BATCH PROCESS (WITH ERROR HANDLING)
    elif page == "📂 Batch Analysis":
        st.subheader("Mass CSV Scanner")
        
        # Sample Download for User Guidance
        sample_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Required Template", sample_csv, "template.csv", "text/csv")
        
        file = st.file_uploader("Upload CSV File", type=["csv"])
        if file:
            try:
                data = pd.read_csv(file)
                # Validation Logic
                missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
                if missing:
                    st.error(f"❌ Galat Data! Missing Columns: {', '.join(missing)}")
                else:
                    with st.status("Analyzing...") as s:
                        probs = pipeline.predict_proba(data[REQUIRED_COLUMNS])[:,1]
                        data["Risk_Score (%)"] = (probs * 100).round(2)
                        data["Final_Status"] = ["LEAVING" if p >= threshold else "STAYING" for p in probs]
                        s.update(label="Analysis Done!", state="complete")
                    
                    # Download Processed File
                    out_csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results", out_csv, "results.csv", "text/csv")
                    st.dataframe(data.style.background_gradient(subset=['Risk_Score (%)'], cmap='RdYlGn_r'))
            except Exception as e:
                st.error(f"🚨 File kharab hai: {e}")

    # PAGE 3: HISTORY
    elif page == "📜 History":
        st.subheader("Recent Activity")
        if st.session_state.history: st.table(st.session_state.history)
        else: st.info("No records.")

    st.markdown("<div style='text-align:center; margin-top:50px; opacity:0.3;'>PRAJWAL RAJPUT | 2026</div>", unsafe_allow_html=True)

# ================= RUNNER =================
if st.session_state.logged_in: main_app()
else: login_page()
