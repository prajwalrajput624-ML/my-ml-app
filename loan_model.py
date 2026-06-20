import pandas as pd
import numpy as np
import streamlit as st
import joblib
import time

st.set_page_config(
    page_title='Loan Approval System',
    page_icon='💰',
    layout='wide'
)

@st.cache_resource()
def load_my_model():
    try:
        return joblib.load('final_loan_model_GB.pkl')
    except Exception as e:
        st.error(f"Model Path Error: {e}")
        return None
    
model = load_my_model()

st.sidebar.title('💻 Navigation')
st.sidebar.divider()

st.sidebar.header('⚙️ Risk Controls')

risk_threshold = st.sidebar.slider(
    label="Max Acceptable Default Risk (%)", 
    min_value=5.0, 
    max_value=95.0, 
    value=50.0, 
    step=1.0,
    help="Applications exceeding this model risk percentage will automatically trigger a policy rejection flag."
)

st.sidebar.divider()
st.sidebar.header('System Info')
if model is not None:
    st.sidebar.success("Model: Ready 🟢")
else:
    st.sidebar.error("Model: Offline 🔴")
    st.sidebar.warning("Please check if 'final_loan_model_GB.pkl' is in the root directory.")

st.markdown("""
    <style>
    div.stButton > button, div.stButton > button:first-child, div.stButton > button[kind="primary"] {
        font-size: 28px !important;   
        font-weight: bold !important;
        height: 65px !important;     
        width: 300px !important;
        border-radius: 12px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div.stButton > button p {
        font-size: 20px !important;
        font-weight: bold !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Loan Approval System")
st.divider()

mode_tab1, mode_tab2, mode_tab3 = st.tabs([
    "👤 Manual Single Entry",
    "📁 Bulk CSV Processing",
    "📊 Risk Insights"
])

with mode_tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Enter Your Age", min_value=18, max_value=65, value=25)
        income = st.number_input('Enter Your Income', min_value=10000, max_value=250000, value=50000)
        experience = st.number_input("Enter Your Experience", min_value=0, max_value=25, value=3)

    with col2:
        home_ownership = st.selectbox("Choose The Home Ownership", ['Rent', 'Mortgage', 'Own', 'Other'])
        loan_intent = st.selectbox("Loan Intent", ['Education', 'Medical', 'Venture', 'Personal', 'Debtconsolidation', 'HomeImprovement'])
        age_category = st.selectbox('Choose The Age Category', ['18-25', '26-35', '36-45', '46-60', '60+'])

    with col3:
        loan_percent_income = st.number_input("Enter The Loan Percentage Income", min_value=0.0, max_value=1.0, step=0.01, value=0.2)
        credit_history_length = st.number_input("Enter Your Credit History Length", min_value=0, max_value=40, value=5)
        loan_amount = st.number_input("Enter Your Loan Amount", min_value=1000, max_value=200000, value=10000)

    loan_int_rate = st.slider(label="Choose Loan Interest Rate", min_value=5.0, max_value=25.0, step=0.1, value=11.0)
    credit_score_category = st.selectbox("Choose The Credit Score Category", ['Low', 'Moderate', 'High'])
    credit_score = st.slider(label="Choose Credit Score", min_value=300, max_value=850, value=650)

    data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_emp_exp': [experience],
        'loan_amnt': [loan_amount],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [credit_history_length],
        'credit_score': [credit_score],
        'person_home_ownership': [home_ownership],
        'loan_intent': [loan_intent],
        'person_age_category': [age_category],
        'credit_score_category': [credit_score_category],
        'loan_int_rate': [loan_int_rate]
    })

    if st.button('Predict', type='primary'):
        if model is not None:
            with st.status("Analyzing profile...", expanded=True) as status:
                time.sleep(1.0)

                probabilities = model.predict_proba(data)
                prob_rejected = probabilities[0][1] * 100
                prob_approved = probabilities[0][0] * 100
                final_decision = 1 if prob_rejected >= risk_threshold else 0

                status.update(label="Analysis Complete!", state='complete', expanded=False)

                if final_decision == 0:
                    st.balloons()
                    st.success('🎉 **Loan Approved!**')
                else:
                    st.error('❌ **Loan Rejected.**')

# ─────────────────────────────────────────────
# TAB 2 — Bulk CSV Processing
# ─────────────────────────────────────────────
with mode_tab2:
    st.markdown("### 📊 Batch Evaluation Processing Engine")
    st.caption("Upload a structured dataset in CSV format matching model parameter keys to pipeline multi-record assessments.")

    with st.expander("CSV Required Column Format"):
        st.write("Your input data file columns must match exactly: `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`, `person_home_ownership`, `loan_intent`, `person_age_category`, `credit_score_category`, `loan_int_rate`")

    uploaded_file = st.file_uploader("Choose a CSV File containing application records", type=["csv"])

    if uploaded_file is not None:
        raw_csv_data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully! Previewing target records:")
        st.dataframe(raw_csv_data.head(5))

        if st.button("Execute Bulk Pipeline Predict", type="primary", key="bulk_btn"):
            if model is not None:
                try:
                    with st.status("Evaluating multi-record stream indicators...", expanded=True) as status:

                        required_features = [
                            'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                            'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score',
                            'person_home_ownership', 'loan_intent', 'person_age_category',
                            'credit_score_category', 'loan_int_rate'
                        ]

                        input_batch = raw_csv_data[required_features].copy()
                        bulk_probabilities = model.predict_proba(input_batch)

                        raw_csv_data['Underwriting_Decision'] = np.where(
                            bulk_probabilities[:, 1] * 100 >= risk_threshold,
                            'Rejected',
                            'Approved'
                        )

                        time.sleep(1.5)
                        status.update(label="Bulk Underwriting Finished!", state='complete', expanded=False)

                    st.markdown("### 📊 Bulk Processing Output Matrix")
                    st.dataframe(raw_csv_data)

                    output_csv = raw_csv_data.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📥 Download Annotated Predictions Dataset",
                        data=output_csv,
                        file_name="loan_underwriting_batch_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as csv_err:
                    st.error(f"Processing Error: Ensure your CSV matches feature columns perfectly. Details: {csv_err}")


with mode_tab3:
    st.markdown("## 📊 Loan Risk Insights Dashboard")
    st.caption("Analysis based on 7,987 loan defaulters from the Credit Risk Dataset.")
    st.divider()

    # KPI Metrics
    st.subheader("🔢 Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Defaulters", "7,987")
    k2.metric("Average Age", "26.4 yrs")
    k3.metric("Avg Debt-to-Income", "20%")
    k4.metric("Avg Interest Rate", "12.75%")

    st.divider()

    # Demographics
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("👤 Gender Distribution")
        gender_data = pd.DataFrame({
            "Gender": ["Male", "Female"],
            "Count": [4377, 3610]
        })
        st.bar_chart(gender_data.set_index("Gender"))

    with col_b:
        st.subheader("🎂 Age Bracket Breakdown")
        age_data = pd.DataFrame({
            "Age Group": ["18–25", "26–35", "36–45"],
            "Defaulters": [4134, 3561, 292]
        })
        st.bar_chart(age_data.set_index("Age Group"))

    st.divider()

    # Housing & Education
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("🏠 Home Ownership")
        housing_data = pd.DataFrame({
            "Type": ["Rent", "Mortgage", "Own", "Other"],
            "Count": [6168, 1614, 177, 28]
        })
        st.bar_chart(housing_data.set_index("Type"))

    with col_d:
        st.subheader("🎓 Education Level")
        edu_data = pd.DataFrame({
            "Degree": ["Bachelor's", "High School", "Associate", "Master's", "Doctorate"],
            "Count": [2496, 2092, 2056, 1247, 96]
        })
        st.bar_chart(edu_data.set_index("Degree"))

    st.divider()

    # Loan Intent
    st.subheader("📋 Loan Intent Distribution")
    intent_data = pd.DataFrame({
        "Intent": ["Medical", "Debt Consolidation", "Education", "Personal", "Home Improvement", "Venture"],
        "Accounts": [1901, 1714, 1241, 1219, 1022, 890]
    })
    st.bar_chart(intent_data.set_index("Intent"))

    st.divider()

    # Credit Score
    st.subheader("🎯 Credit Score Distribution Among Defaulters")
    credit_data = pd.DataFrame({
        "Score Range": ["400–600 (Poor/Fair)", "600–800 (Good/Excellent)"],
        "Defaulters": [1995, 5992]
    })
    st.bar_chart(credit_data.set_index("Score Range"))

    st.info("💡 **Insight:** ~75% of defaulters had a 'Good/Excellent' credit score — credit score alone is NOT a reliable risk filter.")

    st.divider()

    # Correlation Table
    st.subheader("📈 Top Positive Correlations with Default Risk")
    corr_data = pd.DataFrame({
        "Feature": ["Loan Percent Income", "Loan Interest Rate", "Loan Amount"],
        "Correlation": [0.38, 0.33, 0.11],
        "Risk Level": ["🚨 Highest", "🚨 High", "⚠️ Medium"],
        "Key Insight": [
            "Debt eating >20% of income → cash flow collapse",
            "Avg 12.8% rate; peaks at 19.5% crush early careers",
            "Large loans only risky when paired with low income"
        ]
    })
    st.dataframe(corr_data, use_container_width=True, hide_index=True)

    st.divider()

    # Clean Sheet Paradox
    st.subheader("🔍 The Clean Sheet Paradox")
    st.warning("⚠️ **100% of the 7,987 defaulters had NO previous loan defaults on file.** Every single default was a first-time event — historical spotless records gave a false sense of security.")