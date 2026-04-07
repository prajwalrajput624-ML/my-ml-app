import streamlit as st
import joblib
import numpy as np
import pandas as pd
import io

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #f8f9fb; }

    .block-container {
        padding-top: 2rem;
        max-width: 780px;
    }

    h1 { font-size: 2rem !important; font-weight: 600 !important; color: #1a1a2e; }
    h3 { font-size: 1rem !important; font-weight: 500 !important; color: #555; }

    .stButton > button {
        background: #1a1a2e;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2.5rem;
        font-size: 1rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #16213e;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(26,26,46,0.25);
    }

    .result-box {
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
        font-family: 'DM Sans', sans-serif;
    }
    .result-exit {
        background: #fff0f0;
        border: 1.5px solid #f87171;
        color: #991b1b;
    }
    .result-stay {
        background: #f0fdf4;
        border: 1.5px solid #4ade80;
        color: #166534;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .result-sub {
        font-size: 0.9rem;
        opacity: 0.75;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label {
        font-size: 0.88rem !important;
        color: #444 !important;
        font-weight: 500 !important;
    }

    .stSelectbox > div > div {
        border-radius: 8px !important;
        border-color: #dde1ea !important;
    }

    .stat-box {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-num {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Note: Ensure the filename matches your actual exported pipeline
    return joblib.load('random_forest_pipeline.pkl')

try:
    pipeline = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏦 Bank Churn Predictor")
st.markdown("### Will this customer exit or stay?")
st.markdown("---")

if not model_loaded:
    st.error("⚠️ Model file `random_forest_pipeline.pkl` not found. Please place it in the same folder as this app.")
    st.stop()


# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧑 Single Prediction", "📂 Batch CSV Upload"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-label">Personal Info</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        card_type = st.selectbox("Card Type", ["Diamond", "Gold", "Silver", "Platinum"])

    st.markdown('<div class="section-label">Financial Profile</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
        balance = st.number_input("Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=500.0)
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=250000.0, value=60000.0, step=1000.0)
    with col5:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

    st.markdown('<div class="section-label">Account Activity</div>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    with col6:
        has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    with col7:
        is_active = st.selectbox("Is Active Member", ["Yes", "No"])
    with col8:
        complain = st.selectbox("Has Complained", ["No", "Yes"])

    col9, col10 = st.columns(2)
    with col9:
        satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
    with col10:
        point_earned = st.number_input("Points Earned", min_value=0, max_value=1000, value=400, step=10)

    st.markdown("---")

    if st.button("🔍 Predict Churn", key="single_predict"):

        input_data = pd.DataFrame([{
            'Geography':          geography.lower(),
            'Gender':             gender.lower(),
            'Card Type':          card_type.lower(),
            'CreditScore':        credit_score,
            'Age':                age,
            'Tenure':             tenure,
            'Balance':            balance,
            'NumOfProducts':      num_products,
            'HasCrCard':          1 if has_cr_card == "Yes" else 0,
            'IsActiveMember':     1 if is_active == "Yes" else 0,
            'EstimatedSalary':    estimated_salary,
            'Complain':           1 if complain == "Yes" else 0,
            'Satisfaction Score': satisfaction_score,
            'Point Earned':       point_earned,
        }])

        prediction = pipeline.predict(input_data)[0]
        proba      = pipeline.predict_proba(input_data)[0]
        confidence = round(max(proba) * 100, 1)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-box result-exit">
                <div class="result-title">⚠️ Customer Will Exit</div>
                <div class="result-sub">Confidence: {confidence}% &nbsp;|&nbsp; This customer is likely to churn</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-stay">
                <div class="result-title">✅ Customer Will Stay</div>
                <div class="result-sub">Confidence: {confidence}% &nbsp;|&nbsp; This customer is likely to remain</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📊 Prediction Probabilities"):
            prob_df = pd.DataFrame({
                'Outcome':     ['Stay (0)', 'Exit (1)'],
                'Probability': [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"]
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Upload
# ════════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("#### Upload a CSV file with multiple customers")
    st.markdown("The CSV must contain these columns:")
    st.code("Geography, Gender, Card Type, CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Complain, Satisfaction Score, Point Earned")

    # ── Sample CSV Download ──
    sample_data = pd.DataFrame([{
        'Geography': 'france', 'Gender': 'male', 'Card Type': 'diamond',
        'CreditScore': 650, 'Age': 35, 'Tenure': 5, 'Balance': 50000.0,
        'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1,
        'EstimatedSalary': 60000.0, 'Complain': 0,
        'Satisfaction Score': 3, 'Point Earned': 400
    }, {
        'Geography': 'germany', 'Gender': 'female', 'Card Type': 'gold',
        'CreditScore': 480, 'Age': 58, 'Tenure': 2, 'Balance': 0.0,
        'NumOfProducts': 4, 'HasCrCard': 0, 'IsActiveMember': 0,
        'EstimatedSalary': 45000.0, 'Complain': 1,
        'Satisfaction Score': 1, 'Point Earned': 100
    }])

    csv_sample = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=csv_sample,
        file_name="sample_customers.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # ── File Uploader ──
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(df)} customers loaded successfully!")
            st.dataframe(df.head(), use_container_width=True)

            required_cols = [
                'Geography', 'Gender', 'Card Type', 'CreditScore', 'Age',
                'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                'IsActiveMember', 'EstimatedSalary', 'Complain',
                'Satisfaction Score', 'Point Earned'
            ]

            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"❌ These columns are missing: {missing}")
            else:
                if st.button("🔍 Predict All Customers", key="batch_predict"):

                    input_df = df[required_cols].copy()

                    # Lowercase string columns for consistency
                    for col in ['Geography', 'Gender', 'Card Type']:
                        input_df[col] = input_df[col].astype(str).str.lower()

                    predictions = pipeline.predict(input_df)
                    probas      = pipeline.predict_proba(input_df)

                    df['Prediction']       = predictions
                    df['Churn Label']      = df['Prediction'].map({0: '✅ Stay', 1: '⚠️ Exit'})
                    df['Stay Probability'] = (probas[:, 0] * 100).round(1).astype(str) + '%'
                    df['Exit Probability'] = (probas[:, 1] * 100).round(1).astype(str) + '%'

                    # ── Summary Stats ──
                    total     = len(df)
                    churned   = int(predictions.sum())
                    staying   = total - churned
                    churn_pct = round(churned / total * 100, 1)

                    st.markdown("### 📊 Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(f'<div class="stat-box"><div class="stat-num">{total}</div><div class="stat-label">Total Customers</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#166534">{staying}</div><div class="stat-label">Will Stay</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#991b1b">{churned}</div><div class="stat-label">Will Exit</div></div>', unsafe_allow_html=True)
                    with c4:
                        st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#991b1b">{churn_pct}%</div><div class="stat-label">Churn Rate</div></div>', unsafe_allow_html=True)

                    st.markdown("### 📋 Predictions")
                    st.dataframe(
                        df[['Churn Label', 'Stay Probability', 'Exit Probability'] + required_cols],
                        use_container_width=True,
                        hide_index=True
                    )

                    # ── Download Results ──
                    result_csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=result_csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")