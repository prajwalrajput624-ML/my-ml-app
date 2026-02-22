import streamlit as st
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# ================= LOGIN =================
USERNAME = "prajwal"
PASSWORD = "prajwal6575"

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

# ================= ANIMATED TYPING =================
def typing_effect(text, speed=0.04):
    placeholder = st.empty()
    out = ""
    for ch in text:
        out += ch
        placeholder.markdown(out)
        time.sleep(speed)

# ================= FAKE REASONING LINES =================
def reasoning_lines(lines, speed=0.3):
    placeholder = st.empty()
    for line in lines:
        placeholder.info(line)
        time.sleep(speed)
    placeholder.empty()  # remove last line

# ================= LOGIN PAGE =================
def login_page():
    st.title("🔐 Login")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        btn = st.form_submit_button("Login")

    if btn:
        if u == USERNAME and p == PASSWORD:
            st.session_state.logged_in = True
            st.success("Welcome Prajwal 🚀")
            time.sleep(0.8)
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

# ================= MAIN APP =================
def app():
    pipeline = load_model()

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider("Threshold", 0.1, 0.9, 0.3, 0.05)

        page = st.radio(
            "Navigation",
            ["🔮 Single Prediction", "📁 CSV Prediction", "📜 History", "🧠 Model Info"]
        )

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.title("📊 Customer Churn Predictor")

    # =====================================================
    # 🔮 SINGLE PREDICTION
    # =====================================================
    if page == "🔮 Single Prediction":
        with st.form("single"):
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
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
                )

            predict = st.form_submit_button("🚀 Predict")

        if predict:
            reasoning_lines([
                "🤖 Analyzing tenure...",
                "📊 Evaluating monthly charges...",
                "🔍 Checking contract type...",
                "💡 Calculating churn probability..."
            ], speed=0.6)

            with st.spinner("Predicting..."):
                time.sleep(1)
                df = pd.DataFrame([{
                    "tenure": tenure,
                    "MonthlyCharges": monthly,
                    "TotalCharges": total,
                    "gender": gender,
                    "SeniorCitizen": 1 if senior == "Yes" else 0,
                    "Dependents": dependents,
                    "PhoneService": phone,
                    "MultipleLines": multiple,
                    "InternetService": internet,
                    "OnlineSecurity": online_sec,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device,
                    "TechSupport": tech,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment
                }])

                prob = pipeline.predict_proba(df)[0][1]
                pred = int(prob >= threshold)

            if pred:
                typing_effect("❌ Customer is likely to CHURN", 0.05)
            else:
                typing_effect("✅ Customer is likely to STAY", 0.05)

            typing_effect(f"📊 Churn Probability: {round(prob*100,2)}%", 0.04)

            # Save to history
            st.session_state.history.insert(0, {
                "Result": "CHURN" if pred else "STAY",
                "Probability (%)": round(prob * 100, 2),
                "Tenure": tenure,
                "Monthly Charges": monthly,
                "Contract": contract
            })

    # =====================================================
    # 📁 CSV PREDICTION
    # =====================================================
    elif page == "📁 CSV Prediction":
        st.subheader("📁 Upload CSV for Bulk Prediction")
        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)
            st.success("CSV loaded successfully ✅")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 Predict CSV"):
                reasoning_lines([
                    "🤖 Reading CSV...",
                    "🔍 Calculating probabilities...",
                    "📊 Preparing charts..."
                ], speed=0.5)

                with st.spinner("Processing..."):
                    time.sleep(1.2)
                    probs = pipeline.predict_proba(df)[:, 1]

                df["Churn_Probability"] = (probs * 100).round(2)
                df["Churn_Prediction"] = df["Churn_Probability"].apply(
                    lambda x: "YES" if x >= threshold * 100 else "NO"
                )

                # ===== COUNTS =====
                yes_count = (df["Churn_Prediction"] == "YES").sum()
                no_count = (df["Churn_Prediction"] == "NO").sum()

                # ===== METRICS =====
                c1, c2, c3 = st.columns(3)
                c1.metric("❌ Churn YES", yes_count)
                c2.metric("✅ Churn NO", no_count)
                c3.metric("📦 Total", len(df))

                # ===== PIE CHART =====
                st.subheader("🥧 Churn Distribution (Percentage)")
                labels = ["Churn YES", "Churn NO"]
                sizes = [yes_count, no_count]
                colors = ["#e74c3c", "#2ecc71"]

                fig1, ax1 = plt.subplots()
                ax1.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=colors,
                    explode=(0.05,0),
                    wedgeprops={"edgecolor":"white"}
                )
                ax1.axis("equal")
                st.pyplot(fig1)

                # ===== BAR CHART =====
                st.subheader("📊 Churn Count Bar Chart")
                chart_df = pd.DataFrame({
                    "Churn Status": ["YES", "NO"],
                    "Customers": [yes_count, no_count]
                })
                st.bar_chart(chart_df.set_index("Churn Status"))

                # ===== DATAFRAME + DOWNLOAD =====
                st.subheader("📄 Prediction Results")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Result CSV",
                    csv,
                    "churn_predictions.csv",
                    "text/csv"
                )

    # =====================================================
    # 📜 HISTORY
    # =====================================================
    elif page == "📜 History":
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        else:
            st.info("No history yet.")

    # =====================================================
    # 🧠 MODEL INFO
    # =====================================================
    else:
        st.markdown("""
        **Model:** ML Pipeline  
        **Charts:** Pie + Bar (Red = Churn, Green = Stay)  
        **Prediction:** Single + CSV  
        """)

# ================= ROUTER =================
if st.session_state.logged_in:
    app()
else:
    login_page()