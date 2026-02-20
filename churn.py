import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "toast" not in st.session_state:
    st.session_state.toast = ""

USERNAME = "prajwal"
PASSWORD = "prajwal6575"

# ================= ULTRA CSS =================
st.markdown("""
<style>
/* Body & Background */
html, body, [class*="css"] {transition: all 0.6s ease;}
body {background:#020617 !important; color:#e5e7eb !important;}
body.light {background:#f9f9f9 !important; color:#111 !important;}
section[data-testid="stSidebar"] {background:#0f172a !important; transition: all 0.6s ease;}
body.light section[data-testid="stSidebar"]{background:#e5e7eb !important;}

/* Card */
.card {background:#0f172a; padding:28px; border-radius:22px; box-shadow:0 25px 60px rgba(0,0,0,0.7); margin-bottom:25px; transition: all 0.4s ease;}
body.light .card {background:#f3f4f6; color:#111;}

/* Animated Gradient Title */
.predict {font-size:32px; font-weight:900; text-align:center; background: linear-gradient(90deg,#6366F1,#8B5CF6,#A78BFA,#6366F1); background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: gradientMove 3s ease infinite;}
@keyframes gradientMove {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}

/* Progress Bar */
.conf-bar {background:#111827; border-radius:12px; overflow:hidden; height:20px; margin-top:10px;}
body.light .conf-bar {background:#d1d5db;}
.conf-bar-fill {height:100%; width:0%; border-radius:12px; background: linear-gradient(90deg,#6366F1,#8B5CF6,#A78BFA,#6366F1); transition: width 1.2s ease-in-out;}

/* History Card Animation */
.history-card {opacity:0; transform:translateY(20px); animation: fadeUp 0.8s forwards;}
@keyframes fadeUp {to{opacity:1; transform:translateY(0);}}

/* Fade-in Page */
.page {opacity:0; animation: fadeIn 0.8s forwards;}
@keyframes fadeIn {to{opacity:1;}}

/* Toast Notification */
.toast {position: fixed; top:-60px; left:50%; transform:translateX(-50%); background:#6366F1; color:#fff; padding:10px 20px; border-radius:10px; font-weight:bold; transition: top 0.5s ease;}
.toast.show {top:20px;}

/* Button Hover */
.stButton>button {font-size:18px; font-weight:bold; background: linear-gradient(90deg,#6366F1,#8B5CF6,#A78BFA); color:white; border:none; border-radius:12px; padding:10px 20px; cursor:pointer; transition: all 0.3s ease;}
.stButton>button:hover {transform:scale(1.05); box-shadow:0 0 20px rgba(99,102,241,0.7);}

/* Sidebar Fade */
section[data-testid="stSidebar"] div{opacity:0; animation: fadeInSidebar 0.8s forwards;}
section[data-testid="stSidebar"] div:nth-child(1){animation-delay:0.1s;}
section[data-testid="stSidebar"] div:nth-child(2){animation-delay:0.2s;}
section[data-testid="stSidebar"] div:nth-child(3){animation-delay:0.3s;}
@keyframes fadeInSidebar { to {opacity:1;} }
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("churn_model_deployment.pkl","rb") as f:
        return pickle.load(f)

# ================= LOGIN PAGE =================
def login_page():
    st.markdown("<h1 style='text-align:center;'>🔐 Login</h1>", unsafe_allow_html=True)
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        time.sleep(0.5)
        if u==USERNAME and p==PASSWORD:
            st.session_state.logged_in = True
            st.success("Welcome Prajwal 🚀")
            time.sleep(0.3)
            st.rerun()
        else:
            st.error("Wrong credentials ❌")

# ================= MAIN APP =================
def app():
    pipeline = load_model()
    threshold = 0.4  # recall-focused threshold

    # Dark Mode Toggle
    toggle = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = toggle
    st.markdown(f"<body class='{'light' if not toggle else ''}'>", unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.title("⚙️ Navigation")
        page = st.radio("Go to", ["🔮 Prediction","📜 History","🧠 Model Info"])
        if st.button("🚪 Logout"):
            st.session_state.logged_in=False
            st.rerun()

    st.markdown("<h1 style='text-align:center;'>📊 IBM Churn Predictor</h1>", unsafe_allow_html=True)
    st.markdown('<div class="page">', unsafe_allow_html=True)

    # ===== PREDICTION PAGE =====
    if page=="🔮 Prediction":
        with st.form("prediction_form"):
            st.markdown("### Enter Customer Details")
            c1, c2 = st.columns(2)
            with c1:
                tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)
                monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
                total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
                gender = st.selectbox("Gender", ["Female","Male"])
                senior_citizen = st.selectbox("Senior Citizen", [0,1])
                contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
                dependents = st.selectbox("Dependents", ["Yes","No"])
            with c2:
                internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
                payment_method = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer","Credit card"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes","No"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes","No"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes","No"])
                device_protection = st.selectbox("Device Protection", ["Yes","No"])
                online_security = st.selectbox("Online Security", ["Yes","No"])
                online_backup = st.selectbox("Online Backup", ["Yes","No"])
                tech_support = st.selectbox("Tech Support", ["Yes","No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes","No"])
                phone_service = st.selectbox("Phone Service", ["Yes","No"])

            submit = st.form_submit_button("🚀 Predict Churn")

        if submit:
            with st.spinner("Predicting..."): time.sleep(1)
            input_dict = {
                "tenure":[tenure], "MonthlyCharges":[monthly_charges], "TotalCharges":[total_charges],
                "gender":[gender], "SeniorCitizen":[senior_citizen], "Contract":[contract], "Dependents":[dependents],
                "InternetService":[internet_service], "PaymentMethod":[payment_method], "PaperlessBilling":[paperless_billing],
                "StreamingMovies":[streaming_movies], "StreamingTV":[streaming_tv], "DeviceProtection":[device_protection],
                "OnlineSecurity":[online_security], "OnlineBackup":[online_backup], "TechSupport":[tech_support],
                "MultipleLines":[multiple_lines], "PhoneService":[phone_service]
            }
            df = pd.DataFrame(input_dict)
            prob = pipeline.predict_proba(df)[:,1][0]
            pred = int(prob >= threshold)
            st.session_state.history.insert(0, {"Prediction":"Yes" if pred else "No", "Probability":round(prob*100,2)})

            # Animated Prediction Card
            st.markdown(f"""
            <div class="card">
                <div class="predict">{'YES' if pred else 'NO'}</div>
                <div class="conf-bar">
                    <div class="conf-bar-fill" style="width:{prob*100}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.toast = f"Churn Prediction: {'YES' if pred else 'NO'}"

    # ===== HISTORY PAGE =====
    elif page=="📜 History":
        df_hist = pd.DataFrame(st.session_state.history)
        if not df_hist.empty:
            for i,h in enumerate(df_hist.to_dict("records")):
                st.markdown(f"""
                <div class="card history-card" style="animation-delay:{i*0.1}s;">
                    Prediction: {h['Prediction']}<br>
                    Probability: {h['Probability']}%
                    <div class="conf-bar">
                        <div class="conf-bar-fill" style="width:{h['Probability']}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No predictions yet.")

    # ===== MODEL INFO =====
    else:
        st.markdown("""
        <div class="card">
            ✔ Recall-focused Churn Model<br>
            ✔ ROC-AUC ~0.84 (K-Fold CV)<br>
            ✔ Scikit-learn Pipeline + Pickle Deployment<br>
            ✔ 18 Feature Input
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # Close page div

    # ===== FOOTER =====
    st.markdown("<div class='footer'><span>© 2026 Developed | by Prajwal Rajput</span></div>", unsafe_allow_html=True)

    # ===== TOAST =====
    if st.session_state.toast:
        st.markdown(f"<div class='toast show'>{st.session_state.toast}</div>", unsafe_allow_html=True)
        time.sleep(2)
        st.session_state.toast = ""

# ================= ROUTER =================
if st.session_state.logged_in:
    app()
else:
    login_page()