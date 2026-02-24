import streamlit as st
import pandas as pd
import pickle
import time
import xgboost as xgb
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# ================= LOGIN CREDENTIALS =================
USERNAME = "prajwal"
PASSWORD = "prajwal6575"

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("banker_churn.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# ================= TYPING EFFECT =================
def type_writer(text, speed=0.025):
    box = st.empty()
    out = ""
    for ch in text:
        out += ch
        box.markdown(out + "▌")
        time.sleep(speed)
    box.markdown(out)

# ================= CSS =================
st.markdown("""
<style>
.result-churn {
    background:#3d0000;
    padding:15px;
    border-left:6px solid #ff4b4b;
    border-radius:10px;
    color:#ff4b4b;
    font-size:18px;
    font-weight:bold;
}
.result-stay {
    background:#003d1f;
    padding:15px;
    border-left:6px solid #00ff99;
    border-radius:10px;
    color:#00ff99;
    font-size:18px;
    font-weight:bold;
}
.footer {
    text-align:center;
    color:#888;
    margin-top:40px;
    padding:15px;
    font-size:13px;
    border-top:1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN PAGE =================
def login_page():
    st.title("🔐 Secure Login")

    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        btn = st.form_submit_button("Login")

    if btn:
        if u == USERNAME and p == PASSWORD:
            st.session_state.logged_in = True
            st.success("Welcome Prajwal 🚀")
            time.sleep(0.6)
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

# ================= MAIN APP =================
def main_app():

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("⚙️ Settings")

        threshold = st.slider(
            "Sensitivity (Risk Tolerance)",
            0.1, 0.9, 0.3, 0.05,
            help="Set the risk threshold. 0.3 is recommended for banking to catch most potential churners."
        )
        st.markdown(f"**Current Threshold:** `{threshold}`")

        page = st.radio(
            "Navigation",
            ["🔮 Check One Customer", "📂 Batch Analysis (CSV)", "📜 History", "🧠 Model Info"]
        )

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.title("📊 Bank Customer Churn Predictor")

    # ================= SINGLE PREDICTION =================
    if page == "🔮 Check One Customer":
        st.subheader("🧾 Enter Customer Information")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 👤 Personal Profile")
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Education_Level = st.selectbox(
                "Education Level",
                ["Uneducated","High School","College","Graduate","Post-Graduate","Doctorate"]
            )
            Marital_Status = st.selectbox(
                "Marital Status", ["Single","Married","Divorced"]
            )
            Income_Category = st.selectbox(
                "Yearly Income Range",
                ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +"]
            )
            Card_Category = st.selectbox(
                "Credit Card Type", ["Blue","Silver","Gold","Platinum"]
            )

        with c2:
            st.markdown("### 🏦 Banking Activity")
            Customer_Age = st.number_input("Age", 18, 100, 35)
            Dependent_count = st.number_input("Number of Dependents", 0, 10, 2)
            Months_on_book = st.number_input("Tenure with Bank (Months)", 1, 100, 24)
            Total_Relationship_Count = st.number_input("Total Products Held (Accounts/Loans)", 1, 10, 4)
            Months_Inactive_12_mon = st.number_input("Months Inactive (Last 12 Months)", 0, 12, 2)

        st.divider()
        st.markdown("### 💸 Transaction Behavior")
        c3, c4 = st.columns(2)
        
        with c3:
            Contacts_Count_12_mon = st.number_input("Bank Contacts (Last 12 Months)", 0, 12, 2)
            Credit_Limit = st.number_input("Total Credit Limit", 500.0, 50000.0, 10000.0)
            Total_Revolving_Bal = st.number_input("Current Unpaid Balance (Revolving)", 0.0, 50000.0, 1500.0)
        
        with c4:
            Total_Trans_Ct = st.number_input("Total Transaction Count", 1, 300, 60)
            Total_Ct_Chng_Q4_Q1 = st.number_input("Transaction Trend (Q4 vs Q1)", 0.0, 5.0, 1.2, help="Values < 1.0 indicate decreasing usage.")
            Avg_Utilization_Ratio = st.number_input("Credit Card Utilization Rate (0 to 1)", 0.0, 1.0, 0.3)

        if st.button("🚀 Analyze Churn Risk"):
            # Map inputs to model columns
            df = pd.DataFrame({
                "Gender":[Gender],
                "Education_Level":[Education_Level],
                "Marital_Status":[Marital_Status],
                "Income_Category":[Income_Category],
                "Card_Category":[Card_Category],
                "Customer_Age":[Customer_Age],
                "Dependent_count":[Dependent_count],
                "Months_on_book":[Months_on_book],
                "Total_Relationship_Count":[Total_Relationship_Count],
                "Months_Inactive_12_mon":[Months_Inactive_12_mon],
                "Contacts_Count_12_mon":[Contacts_Count_12_mon],
                "Credit_Limit":[Credit_Limit],
                "Total_Revolving_Bal":[Total_Revolving_Bal],
                "Total_Trans_Ct":[Total_Trans_Ct],
                "Total_Ct_Chng_Q4_Q1":[Total_Ct_Chng_Q4_Q1],
                "Avg_Utilization_Ratio":[Avg_Utilization_Ratio]
            })

            type_writer("🤖 Running AI analysis on behavior patterns...")

            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            # Calculate probabilities and round to 2 decimals
            raw_prob = pipeline.predict_proba(df)[:,1][0]
            prob_percent = round(float(raw_prob) * 100, 2) 
            pred = int(raw_prob >= threshold)

            st.session_state.history.insert(0,{
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Prediction": "⚠️ LEAVING" if pred else "✅ STAYING",
                "Risk Score": f"{prob_percent}%"
            })

            if pred:
                st.markdown(f'<div class="result-churn">❌ HIGH RISK: Customer is likely to CHURN!</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-stay">✅ LOW RISK: Customer is likely to STAY.</div>', unsafe_allow_html=True)

            st.info(f"📊 Risk Probability Score: {prob_percent}%")

    # ================= CSV PREDICTION =================
    elif page == "📂 Batch Analysis (CSV)":
        st.subheader("📂 Bulk Customer Analysis")
        file = st.file_uploader("Upload CSV File", type=["csv"])

        if file:
            df = pd.read_csv(file)

            probs = pipeline.predict_proba(df)[:,1]
            df["Risk_Score (%)"] = (probs * 100).round(2)
            df["Final_Status"] = ["LEAVING" if p >= threshold else "STAYING" for p in probs]

            churn_yes = (probs >= threshold).sum()
            churn_no = (probs < threshold).sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("⚠️ High Risk (Leaving)", churn_yes)
            c2.metric("✅ Low Risk (Staying)", churn_no)
            c3.metric("👥 Total Analyzed", len(df))

            st.markdown("### 📊 Churn Distribution")
            chart_df = pd.DataFrame(
                {"Customers":[churn_yes, churn_no]},
                index=["Leaving","Staying"]
            )
            st.bar_chart(chart_df, height=300)

            st.dataframe(df, use_container_width=True)

    # ================= HISTORY =================
    elif page == "📜 History":
        st.subheader("📜 Recent Prediction History")
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        else:
            st.info("No predictions made yet.")

    # ================= MODEL INFO =================
    else:
        st.subheader("🧠 Model Intelligence")
        st.markdown(f"""
        This system utilizes a **XGBoost Classifier** to analyze banking behavior and predict potential customer exits.
        
        **Model Accuracy:** 92%  
        **ROC-AUC Score:** 0.96  
        **Current Sensitivity:** {threshold}  
        
        ---
        **Key Churn Indicators:**
        * Declining Transaction Counts (Q4 vs Q1)
        * High Number of Bank Contacts
        * Long Periods of Inactivity
        """)

    # ================= FOOTER =================
    st.markdown(
        "<div class='footer'>© 2026 Developed by | Prajwal Rajput</div>",
        unsafe_allow_html=True
    )

# ================= ROUTER =================
if st.session_state.logged_in:
    main_app()
else:
    login_page()
