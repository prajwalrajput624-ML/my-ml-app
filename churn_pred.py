import streamlit as st
import pandas as pd
import pickle
import time
import xgboost as xgb
from datetime import datetime

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

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("banker_churn.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# ================= ADVANCED ANIMATED CSS =================
st.markdown("""
<style>
    /* Global Background Animation */
    .stApp {
        background: linear-gradient(-45deg, #000428, #004e92, #000000, #020111);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
        color: #e0e0e0;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Input Fields Styling */
    .stNumberInput, .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 5px;
    }

    /* High-Tech Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        background: linear-gradient(45deg, #00dbde, #fc00ff);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        padding: 10px;
        transition: 0.5s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(252, 0, 255, 0.5);
        color: #fff;
    }

    /* Result Card Animations */
    .result-card {
        padding: 30px;
        border-radius: 25px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        margin-top: 20px;
        animation: zoomIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    @keyframes zoomIn {
        from { opacity: 0; transform: scale(0.5); }
        to { opacity: 1; transform: scale(1); }
    }

    .churn-alert {
        background: rgba(255, 0, 0, 0.1);
        border: 2px solid #ff4b4b;
        box-shadow: 0 0 30px rgba(255, 75, 75, 0.3);
    }

    .stay-success {
        background: rgba(0, 255, 127, 0.1);
        border: 2px solid #00ff7f;
        box-shadow: 0 0 30px rgba(0, 255, 127, 0.3);
    }

    /* Pulse for Headlines */
    .glitch-text {
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { text-shadow: 0 0 5px #00dbde; }
        50% { text-shadow: 0 0 20px #fc00ff; }
        100% { text-shadow: 0 0 5px #00dbde; }
    }
</style>
""", unsafe_allow_html=True)

# ================= LOGIN PAGE =================
def login_page():
    st.markdown("<br><br><h1 style='text-align: center;' class='glitch-text'>🛰️ AI COMMAND CENTER</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1.5,1])
    with col2:
        st.markdown("<div style='background:rgba(255,255,255,0.05); padding:30px; border-radius:20px;'>", unsafe_allow_html=True)
        with st.form("login_form"):
            u = st.text_input("Access Key (Username)")
            p = st.text_input("Security Code (Password)", type="password")
            login_btn = st.form_submit_button("INITIALIZE SYSTEM")
        st.markdown("</div>", unsafe_allow_html=True)

        if login_btn:
            if u == "prajwal" and p == "prajwal6575":
                st.session_state.logged_in = True
                st.toast("System Initialized. Welcome, Commander.", icon="🚀")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Access Denied: Invalid Credentials")

# ================= MAIN APP =================
def main_app():
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='glitch-text'>Core Engine</h2>", unsafe_allow_html=True)
        threshold = st.sidebar.select_slider(
            "Risk Sensitivity",
            options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            value=0.3
        )
        
        st.divider()
        nav = st.radio("Navigation", ["🔮 Risk Scanner", "📂 Data Batcher", "📜 System Logs"])
        
        if st.button("🚪 TERMINATE SESSION"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown(f"<h1>⚡ Neural Churn Analysis <small style='color:#00dbde;'>v4.0</small></h1>", unsafe_allow_html=True)

    if nav == "🔮 Risk Scanner":
        st.markdown("### 📝 Input Customer Parameters")
        
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                Gender = st.selectbox("Gender", ["Male", "Female"])
                Education = st.selectbox("Education", ["Uneducated","High School","College","Graduate","Post-Graduate","Doctorate"])
                Age = st.number_input("Customer Age", 18, 100, 35)
                Dependents = st.number_input("Dependents", 0, 10, 2)
            
            with c2:
                Income = st.selectbox("Annual Income", ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +"])
                Card = st.selectbox("Card Category", ["Blue","Silver","Gold","Platinum"])
                Tenure = st.number_input("Tenure (Months)", 1, 100, 24)
                Products = st.number_input("Products Used", 1, 10, 4)
            
            with c3:
                Inactive = st.number_input("Months Inactive", 0, 12, 2)
                Contacts = st.number_input("Bank Contacts", 0, 12, 2)
                Limit = st.number_input("Credit Limit", 500, 50000, 10000)
                Revolving = st.number_input("Current Balance", 0, 50000, 1500)

        st.divider()
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            Trans_Ct = st.number_input("Transaction Count", 1, 300, 60)
        with cc2:
            Trend = st.number_input("Usage Trend (Q4/Q1)", 0.0, 5.0, 1.2)
        with cc3:
            Utilization = st.slider("Utilization Rate", 0.0, 1.0, 0.3)

        if st.button("🔥 RUN PREDICTIVE SCAN"):
            df = pd.DataFrame({
                "Gender":[Gender], "Education_Level":[Education], "Marital_Status":["Single"],
                "Income_Category":[Income], "Card_Category":[Card], "Customer_Age":[Age],
                "Dependent_count":[Dependents], "Months_on_book":[Tenure], "Total_Relationship_Count":[Products],
                "Months_Inactive_12_mon":[Inactive], "Contacts_Count_12_mon":[Contacts], "Credit_Limit":[Limit],
                "Total_Revolving_Bal":[Revolving], "Total_Trans_Ct":[Trans_Ct], "Total_Ct_Chng_Q4_Q1":[Trend],
                "Avg_Utilization_Ratio":[Utilization]
            })

            with st.status("🧠 Scanning Neural Networks...", expanded=True) as status:
                time.sleep(1)
                st.write("Extracting behavioral features...")
                time.sleep(0.8)
                st.write("Calculating probability vectors...")
                raw_prob = pipeline.predict_proba(df)[:,1][0]
                prob_percent = round(float(raw_prob) * 100, 2)
                pred = int(raw_prob >= threshold)
                status.update(label="Scan Complete!", state="complete")

            # ANIMATED FEEDBACK
            if pred:
                st.snow() # Animated Snow for Alert
                st.markdown(f"""
                <div class='result-card churn-alert'>
                    <h1 style='color:#ff4b4b;'>🚨 CRITICAL RISK 🚨</h1>
                    <h2 style='margin:0;'>PROBABILITY: {prob_percent}%</h2>
                    <p style='font-size:18px;'>This customer is highly likely to leave. Deploy retention strategy immediately.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.balloons() # Animated Balloons
                st.confetti = True # Conceptually trigger jashn
                st.markdown(f"""
                <div class='result-card stay-success'>
                    <h1 style='color:#00ff7f;'>💎 LOYAL CUSTOMER 💎</h1>
                    <h2 style='margin:0;'>PROBABILITY: {prob_percent}%</h2>
                    <p style='font-size:18px;'>Customer activity is stable. No churn risk detected at current threshold.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.history.insert(0, {"Time": datetime.now().strftime("%H:%M:%S"), "Status": "🚨 RISK" if pred else "✅ SAFE", "Score": f"{prob_percent}%"})

    elif nav == "📂 Data Batcher":
        st.subheader("📡 Mass Intelligence Processing")
        file = st.file_uploader("Upload Target CSV", type="csv")
        if file:
            input_df = pd.read_csv(file)
            with st.spinner("Decoding Batch Data..."):
                time.sleep(2)
                batch_probs = pipeline.predict_proba(input_df)[:,1]
                input_df["Risk_Score (%)"] = (batch_probs * 100).round(2)
                input_df["Action"] = ["RETENTION REQUIRED" if p >= threshold else "MAINTAIN" for p in batch_probs]
            
            st.dataframe(input_df.style.background_gradient(subset=['Risk_Score (%)'], cmap='coolwarm'), use_container_width=True)

    elif nav == "📜 System Logs":
        st.subheader("📑 Historical Archives")
        if st.session_state.history:
            st.table(pd.DataFrame(st.session_state.history))
        else:
            st.info("No logs found in the current session.")

    st.markdown(f"<div style='text-align:center; margin-top:50px; opacity:0.5; font-size:12px;'>SYSTEM CORE: XGBOOST | ENCRYPTED BY PRAJWAL RAJPUT | {datetime.now().year}</div>", unsafe_allow_html=True)

# ================= ROUTER =================
if st.session_state.logged_in:
    main_app()
else:
    login_page()
