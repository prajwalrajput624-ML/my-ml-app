import streamlit as st
import pandas as pd
import pickle
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="IBM Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []

USERNAME = "prajwal"
PASSWORD = "prajwal6575"

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

# ================= CSS =================
st.markdown("""
<style>
.prob-box {
    background:#0e1117;
    padding:20px;
    border-radius:14px;
    box-shadow:0 0 18px rgba(0,255,180,0.35);
    margin-top:20px;
}
.bar {
    height:20px;
    width:100%;
    background:#222;
    border-radius:12px;
    overflow:hidden;
}
.fill {
    height:100%;
    width:0%;
    background:linear-gradient(90deg,#00ffcc,#00bfff);
    animation: load 1.5s forwards;
}
@keyframes load {
    to { width: var(--val); }
}
.footer {
    text-align:center;
    color:gray;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN PAGE =================
def login_page():
    st.title("🔐 Login")

    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        login = st.form_submit_button("Login")

    if login:
        if u == USERNAME and p == PASSWORD:
            st.session_state.logged_in = True
            st.success("Welcome Prajwal 🚀")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

# ================= MAIN APP =================
def app():
    pipeline = load_model()
    threshold = 0.5

    st.title("📊 IBM Customer Churn Predictor")

    with st.sidebar:
        page = st.radio(
            "Navigation",
            ["🔮 Prediction", "📜 History", "🧠 Model Info"]
        )
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # ================= PREDICTION =================
    if page == "🔮 Prediction":
        with st.form("predict"):
            st.subheader("Enter Customer Details")

            c1, c2 = st.columns(2)

            with c1:
                tenure = st.number_input("Tenure", 0, 100, 12)
                monthly = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
                total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior Citizen", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multiple = st.selectbox("Multiple Lines", ["Yes", "No"])

            with c2:
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_sec = st.selectbox("Online Security", ["Yes", "No"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No"])
                device = st.selectbox("Device Protection", ["Yes", "No"])
                tech = st.selectbox("Tech Support", ["Yes", "No"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
                contract = st.selectbox(
                    "Contract", ["Month-to-month", "One year", "Two year"]
                )
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
                )

            predict = st.form_submit_button("🚀 Predict Churn")

        if predict:
            with st.spinner("Predicting..."):
                time.sleep(1)

            df = pd.DataFrame({
                "tenure": [tenure],
                "MonthlyCharges": [monthly],
                "TotalCharges": [total],
                "gender": [gender],
                "SeniorCitizen": [1 if senior == "Yes" else 0],
                "Dependents": [dependents],
                "PhoneService": [phone],
                "MultipleLines": [multiple],
                "InternetService": [internet],
                "OnlineSecurity": [online_sec],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device],
                "TechSupport": [tech],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless],
                "PaymentMethod": [payment]
            })

            prob = pipeline.predict_proba(df)[:, 1][0]
            pred = int(prob >= threshold)

            st.subheader("🔮 Prediction Result")

            if pred == 1:
                st.error("❌ Customer is likely to CHURN")
            else:
                st.success("✅ Customer is likely to STAY")

            st.info(f"📊 Churn Probability: {round(prob*100,2)}%")

            st.markdown(f"""
            <div class="prob-box">
                <b>Churn Probability</b>
                <div class="bar">
                    <div class="fill" style="--val:{prob*100}%"></div>
                </div>
                <p>{round(prob*100,2)}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.history.insert(
                0,
                {"Result": "CHURN" if pred else "STAY", "Probability": round(prob*100, 2)}
            )

    # ================= HISTORY =================
    elif page == "📜 History":
        st.subheader("📜 Prediction History")
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history))
        else:
            st.info("No predictions yet.")

    # ================= MODEL INFO =================
    else:
        st.subheader("🧠 Model Information")
        st.markdown("""
        **Model Used:** GradientBoostingClassifier  
        **Evaluation Metric:** ROC-AUC  
        **ROC-AUC Score:** **0.8432612570719986**
        """)

    st.markdown("<div class='footer'>© 2026 Developed by | Prajwal Rajput</div>", unsafe_allow_html=True)

# ================= ROUTER =================
if st.session_state.logged_in:
    app()
else:
    login_page()