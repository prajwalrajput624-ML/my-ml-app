import streamlit as st
import pandas as pd
import pickle
import time
import xgboost as xgb
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Churn Predictor",
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
    with open("banker_churn.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# ================= ADVANCED ANIMATED CSS =================
st.markdown("""
<style>
    /* Animated Gradient Background */
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

    /* Glassmorphism Effect for Sidebar and Widgets */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #ff8c00, #ff0080);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s all ease;
        transform: scale(1);
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 20px rgba(255, 0, 128, 0.6);
    }

    /* Animated Result Cards */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        animation: slideIn 0.8s ease-out;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .churn-box {
        background: rgba(255, 75, 75, 0.15);
        border-left: 10px solid #ff4b4b;
    }

    .stay-box {
        background: rgba(0, 255, 153, 0.15);
        border-left: 10px solid #00ff99;
    }

    /* Simple Pulse Animation for Text */
    .pulse {
        animation: pulse-animation 2s infinite;
    }

    @keyframes pulse-animation {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ================= LOGIN PAGE =================
def login_page():
    st.markdown("<h1 style='text-align: center; color: white;'>🔐 Secure AI Portal</h1>", unsafe_allow_html=True)
    
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
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='pulse'>⚙️ Control Center</h2>", unsafe_allow_html=True)
        threshold = st.slider("Model Sensitivity", 0.1, 0.9, 0.3, 0.05)
        
        page = st.selectbox("Navigate To", ["🔮 Predictor", "📂 Batch Process", "📜 Logs"])
        
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown(f"<h1>🚀 AI Customer Insights <small style='font-size:15px; color:#bbb;'>v3.0</small></h1>", unsafe_allow_html=True)

    if page == "🔮 Predictor":
        st.subheader("Customer Data Entry")
        
        # Using columns for layout
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                Gender = st.selectbox("Gender", ["Male", "Female"])
                Education = st.selectbox("Education", ["Uneducated","High School","College","Graduate","Post-Graduate","Doctorate"])
                Income = st.selectbox("Annual Income", ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +"])
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
        with col3:
            Trans_Ct = st.number_input("Total Transactions", 1, 300, 60)
        with col4:
            Trend = st.number_input("Activity Trend (Q4/Q1)", 0.0, 5.0, 1.2)
        
        Utilization = st.slider("Credit Utilization Rate", 0.0, 1.0, 0.3)

        if st.button("✨ Analyze Risk Now"):
            df = pd.DataFrame({
                "Gender":[Gender], "Education_Level":[Education], "Marital_Status":["Single"],
                "Income_Category":[Income], "Card_Category":[Card], "Customer_Age":[Age],
                "Dependent_count":[Dependents], "Months_on_book":[Tenure], "Total_Relationship_Count":[Products],
                "Months_Inactive_12_mon":[Inactive], "Contacts_Count_12_mon":[Contacts], "Credit_Limit":[Limit],
                "Total_Revolving_Bal":[Revolving], "Total_Trans_Ct":[Trans_Ct], "Total_Ct_Chng_Q4_Q1":[Trend],
                "Avg_Utilization_Ratio":[Utilization]
            })

            with st.spinner("🤖 AI Thinking..."):
                time.sleep(1.5)
                raw_prob = pipeline.predict_proba(df)[:,1][0]
                prob_percent = round(float(raw_prob) * 100, 2)
                pred = int(raw_prob >= threshold)

            # Animated Results
            if pred:
                st.markdown(f"""
                <div class='result-card churn-box'>
                    <h2 style='color:#ff4b4b;'>⚠️ HIGH RISK DETECTED</h2>
                    <p>There is a <b>{prob_percent}%</b> chance this customer will leave.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-card stay-box'>
                    <h2 style='color:#00ff99;'>✅ CUSTOMER IS LOYAL</h2>
                    <p>Churn probability is only <b>{prob_percent}%</b>. Everything looks good!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # History log
            st.session_state.history.insert(0, {"Time": datetime.now().strftime("%H:%M:%S"), "Status": "Risk" if pred else "Safe", "Score": f"{prob_percent}%"})

    elif page == "📂 Batch Process":
        st.subheader("CSV Intelligence Analysis")
        uploaded_file = st.file_uploader("Upload bank data...", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            with st.status("Processing Data...", expanded=True) as status:
                st.write("Reading file...")
                time.sleep(1)
                st.write("Running XGBoost predictions...")
                probs = pipeline.predict_proba(data)[:,1]
                data["Risk_Score (%)"] = (probs * 100).round(2)
                data["Prediction"] = ["LEAVING" if p >= threshold else "STAYING" for p in probs]
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            st.dataframe(data.style.background_gradient(subset=['Risk_Score (%)'], cmap='Reds'))

    elif page == "📜 Logs":
        st.subheader("Recent Activity Logs")
        if st.session_state.history:
            st.table(pd.DataFrame(st.session_state.history))
        else:
            st.info("No activity recorded yet.")

    # Footer
    st.markdown(f"<div class='footer'>AI Core: XGBoost | Developed by Prajwal Rajput | {datetime.now().year}</div>", unsafe_allow_html=True)

# ================= ROUTER =================
if st.session_state.logged_in:
    main_app()
else:
    login_page()
