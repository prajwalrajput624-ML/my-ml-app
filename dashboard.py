import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import joblib

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="AI Churn Platform",
    layout="wide"
)

# ================= DARK UI =================

st.markdown("""
<style>
.stApp{
background-color:#0b0f19;
color:white;
}

section[data-testid="stSidebar"]{
background:#111827;
}

.stButton>button{
background:#2563eb;
color:white;
border-radius:8px;
padding:8px 16px;
}
</style>
""", unsafe_allow_html=True)

# ================= DATABASE =================

conn = sqlite3.connect("users.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT UNIQUE,
password TEXT
)
""")

conn.commit()

# ================= PASSWORD HASH =================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ================= REGISTER =================

def register_user(username,password):

    try:
        cur.execute(
        "INSERT INTO users(username,password) VALUES(?,?)",
        (username,hash_password(password))
        )
        conn.commit()
        return True
    except:
        return False

# ================= LOGIN =================

def login_user(username,password):

    cur.execute(
    "SELECT * FROM users WHERE username=? AND password=?",
    (username,hash_password(password))
    )

    return cur.fetchone()

# ================= LOAD MODEL =================

model = joblib.load("random_forest_pipeline.pkl")

# ================= SESSION =================

if "user" not in st.session_state:
    st.session_state.user = None

# ================= LOGIN PAGE =================

def login_page():

    st.title("AI Customer Churn Platform")

    tab1,tab2 = st.tabs(["Login","Register"])

    with tab1:

        username = st.text_input("Username")
        password = st.text_input("Password",type="password")

        if st.button("Login"):

            user = login_user(username,password)

            if user:
                st.session_state.user=username
                st.rerun()
            else:
                st.error("Invalid login")

    with tab2:

        username = st.text_input("New Username")
        password = st.text_input("New Password",type="password")

        if st.button("Create Account"):

            if register_user(username,password):
                st.success("Account created")
            else:
                st.error("Username already exists")

# ================= REQUIRE LOGIN =================

if st.session_state.user is None:
    login_page()
    st.stop()

# ================= SIDEBAR =================

st.sidebar.title("AI Churn App")
st.sidebar.write("User:",st.session_state.user)

page = st.sidebar.radio(
"Navigation",
[
"Dashboard",
"Predict Churn"
]
)

# ================= DASHBOARD =================

if page=="Dashboard":

    st.title("Customer Churn Dashboard")

    c1,c2,c3 = st.columns(3)

    c1.metric("Total Customers","10000")
    c2.metric("Churn Rate","20.3%")
    c3.metric("Model","Random Forest")

# ================= PREDICTION =================

if page=="Predict Churn":

    st.title("Customer Churn Prediction")

    # categorical

    geography = st.selectbox("Geography",["France","Germany","Spain"])
    gender = st.selectbox("Gender",["Male","Female"])
    card_type = st.selectbox("Card Type",["Silver","Gold","Platinum","Diamond"])

    # numeric

    credit_score = st.slider("CreditScore",300,850,650)
    age = st.slider("Age",18,90,40)
    tenure = st.slider("Tenure",0,10,3)

    balance = st.number_input("Balance",0.0,300000.0,60000.0)

    products = st.selectbox("NumOfProducts",[1,2,3,4])

    has_card = st.selectbox("HasCrCard",[0,1])
    active = st.selectbox("IsActiveMember",[0,1])

    salary = st.number_input("EstimatedSalary",0.0,200000.0,50000.0)

    complain = st.selectbox("Complain",[0,1])

    satisfaction = st.slider("Satisfaction Score",1,5,3)

    points = st.number_input("Point Earned",0,1000,400)

    if st.button("Predict"):

        data = pd.DataFrame({

        "Geography":[geography],
        "Gender":[gender],
        "Card Type":[card_type],

        "CreditScore":[credit_score],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[products],
        "HasCrCard":[has_card],
        "IsActiveMember":[active],
        "EstimatedSalary":[salary],
        "Complain":[complain],
        "Satisfaction Score":[satisfaction],
        "Point Earned":[points]

        })

        prob = model.predict_proba(data)[0][1]

        if prob > 0.5:
            st.error(f"High churn risk : {prob*100:.2f}%")
        else:
            st.success(f"Low churn risk : {prob*100:.2f}%")