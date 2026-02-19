import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Meal Time Predictor", page_icon="🍽️", layout="wide")

# ================= SESSION =================
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
/* BODY & GENERAL */
html, body, [class*="css"] {background:#020617 !important; color:#e5e7eb !important; transition: all 0.6s ease;}
body.light {background:#f9f9f9 !important; color:#111 !important;}
section[data-testid="stSidebar"] {background:#0f172a !important; animation: slideInLeft 0.8s ease; transition: all 0.6s ease;}
body.light section[data-testid="stSidebar"]{background:#e5e7eb !important;}

/* KEYFRAMES */
@keyframes slideInLeft { from {transform:translateX(-60px); opacity:0;} to {transform:translateX(0); opacity:1;} }
@keyframes fadeUp { from {transform:translateY(50px); opacity:0;} to {transform:translateY(0); opacity:1;} }
@keyframes glow {0%{box-shadow:0 0 10px #6366f1;}50%{box-shadow:0 0 25px #8b5cf6;}100%{box-shadow:0 0 10px #6366f1;} }
@keyframes float {0%{transform:translateY(0);}50%{transform:translateY(-8px);}100%{transform:translateY(0);} }
@keyframes bounce {0%,20%,50%,80%,100% {transform: translateY(0);}40% {transform: translateY(-15px);}60% {transform: translateY(-7px);} }
@keyframes pop {0%{transform:scale(1);}50%{transform:scale(1.1);}100%{transform:scale(1);}}
@keyframes colorFlash {0%{background:#6366F1;}50%{background:#8B5CF6;}100%{background:#6366F1;}}
@keyframes wave {0%{background-position:0% 0;}100%{background-position:100% 0;}}
@keyframes spinEmoji {0%{transform:rotate(0deg);}50%{transform:rotate(20deg);}100%{transform:rotate(0deg);} }
@keyframes sparkle {0%{opacity:0; transform:scale(0.5) rotate(0deg);}50%{opacity:1; transform:scale(1.2) rotate(180deg);}100%{opacity:0; transform:scale(0.5) rotate(360deg);} }

/* CARD */
.card {background:#0f172a; padding:28px; border-radius:22px; box-shadow:0 25px 60px rgba(0,0,0,0.7); animation:fadeUp 0.8s ease; margin-bottom:25px; transition: all 0.4s ease;}
body.light .card {background:#f3f4f6; color:#111;}

/* PREDICTION TEXT */
.predict {
    font-size:32px;
    font-weight:900;
    animation:glow 1.6s infinite, float 3s infinite, bounce 1s;
    text-align:center;
    background: linear-gradient(90deg,#6366F1,#8B5CF6,#A78BFA,#6366F1);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.predict.pop {animation: pop 0.5s ease, wave 3s linear infinite, colorFlash 0.5s ease;}

/* SPIN EMOJI */
.spin-emoji {
    font-size:60px;
    display:block;
    text-align:center;
    animation:spinEmoji 0.8s ease, float 3s infinite;
    margin:10px auto;
}

/* SPARKLE */
.sparkle {
    position:absolute;
    font-size:24px;
    animation:sparkle 0.8s ease forwards;
}

/* BUTTON */
.stButton button {background:linear-gradient(90deg,#6366F1,#8B5CF6) !important; color:white !important; height:56px; border-radius:18px; font-size:18px; transition:0.35s;}
.stButton button:hover {transform:scale(1.12);}
.stButton button:active {transform: scale(0.95);}
input, select {border-radius:14px !important;}

/* CONFIDENCE BAR */
.conf-bar {background:#111827; border-radius:12px; overflow:hidden; height:20px; margin-top:10px;}
.conf-bar-fill {
    height:100%;
    background: linear-gradient(90deg,#6366F1,#8B5CF6,#A78BFA,#6366F1);
    width:0%;
    animation: fillBar 1s forwards, wave 3s linear infinite;
    border-radius:12px;
    background-size: 300% 100%;
}
@keyframes fillBar {from {width:0%;} to {width: var(--val);}}

/* HISTORY CARDS */
.history-card {animation: fadeUp 0.8s ease;}

/* FOOTER */
.footer {text-align:center; margin-top:40px; color:#9ca3af; animation:fadeUp 1s ease; overflow:hidden; white-space:nowrap;}
.footer span {display:inline-block; animation: marquee 12s linear infinite;}
@keyframes marquee {0%{transform:translateX(100%);}100%{transform:translateX(-100%);}}

/* TOAST */
.toast {position:fixed; top:20px; right:20px; background:#6366F1; color:white; padding:15px 25px; border-radius:15px; box-shadow:0 10px 25px rgba(0,0,0,0.3); animation: fadeInOut 2s forwards;}
@keyframes fadeInOut {0%{opacity:0; transform:translateY(-30px);}10%{opacity:1; transform:translateY(0);}90%{opacity:1; transform:translateY(0);}100%{opacity:0; transform:translateY(-30px);}}

@media(max-width:768px){
    h1{font-size:26px !important; text-align:center;}
    .card{padding:18px !important; border-radius:16px !important;}
    .predict{font-size:24px !important; text-align:center;}
    .stButton button{width:100% !important; height:50px; font-size:16px;}
    input, select{width:100% !important;}
    section[data-testid="stSidebar"]{width:230px !important;}
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("tips.pkl","rb") as f:
        return pickle.load(f)

# ================= LOGIN PAGE =================
def login_page():
    st.markdown("<h1 style='text-align:center;animation:fadeUp 1s;'>🔐 Login</h1>", unsafe_allow_html=True)
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        time.sleep(0.8)
        if u==USERNAME and p==PASSWORD:
            st.session_state.logged_in = True
            st.success("Welcome Prajwal 🚀")
            time.sleep(0.4)
            st.rerun()
        else:
            st.error("Wrong credentials ❌")

# ================= MAIN APP =================
def app():
    model = load_model()
    toggle = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = toggle
    st.markdown(f"<body class='{'light' if not toggle else ''}'>", unsafe_allow_html=True)

    with st.sidebar:
        st.title("⚙️ Navigation")
        page = st.radio("Go to", ["🔮 Prediction","📜 History","📊 Insights","🧠 Model Info"])
        if st.button("🚪 Logout"):
            st.session_state.logged_in=False
            st.rerun()

    st.markdown("<h1 style='text-align:center;animation:fadeUp 1s;'>🍽️ Meal Time Predictor</h1>", unsafe_allow_html=True)

    # ===== PREDICTION PAGE =====
    if page=="🔮 Prediction":
        with st.form("prediction_form"):
            c1,c2=st.columns(2)
            with c1:
                bill=st.number_input("Total Bill",0.0,500.0,20.0)
                tip=st.number_input("Tip",0.0,50.0,3.0)
                size=st.slider("Family Size",1,10,2)
            with c2:
                sex=st.selectbox("Gender",["male","female"])
                smoker=st.selectbox("Smoker",["yes","no"])
                day=st.selectbox("Day",["sun","sat","fri","thur"])
            submit=st.form_submit_button("🚀 Predict")

        if submit:
            with st.spinner("Predicting..."): time.sleep(1)
            df=pd.DataFrame([{"total_bill":bill,"tip":tip,"sex":sex,"smoker":smoker,"day":day,"size":size}])
            pred=model.predict(df)[0]
            conf=np.max(model.predict_proba(df))*100
            st.session_state.history.insert(0,{"Prediction":pred,"Confidence":round(conf,2)})

            # Animated prediction card with sparkle
            st.markdown(f"""
            <div class="card">
                <div class="predict pop">🍽️ {pred.upper()}</div>
                <div class="spin-emoji">{'🌙' if pred.lower()=='dinner' else '☀️'}</div>
                <div class="conf-bar">
                    <div class="conf-bar-fill" style="--val:{conf}%"></div>
                </div>
                <div class="sparkle">✨</div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.toast = f"Meal Predicted: {pred.upper()} 🍽️"

    # ===== HISTORY PAGE =====
    elif page=="📜 History":
        df_hist = pd.DataFrame(st.session_state.history)
        if not df_hist.empty:
            col1,col2,col3 = st.columns(3)
            day_filter = col1.selectbox("Filter by Day", options=["All","sun","sat","fri","thur"])
            gender_filter = col2.selectbox("Filter by Gender", options=["All","male","female"])
            smoker_filter = col3.selectbox("Filter by Smoker", options=["All","yes","no"])
            filtered = df_hist.copy()
            if day_filter != "All": filtered = filtered[filtered["Prediction"].str.contains(day_filter, case=False)]
            if gender_filter != "All": filtered = filtered[filtered["Prediction"].str.contains(gender_filter, case=False)]
            if smoker_filter != "All": filtered = filtered[filtered["Prediction"].str.contains(smoker_filter, case=False)]
        else:
            filtered = pd.DataFrame(columns=["Prediction","Confidence"])
        for i,h in enumerate(filtered.to_dict("records")):
            st.markdown(f"""
            <div class="card history-card" style="animation-delay:{i*0.15}s;">
                🍽️ {h['Prediction']}<br>
                Confidence: {h['Confidence']}%
                <div class="conf-bar">
                    <div class="conf-bar-fill" style="--val:{h['Confidence']}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ===== INSIGHTS PAGE =====
    elif page=="📊 Insights":
        df=pd.read_csv("tips.csv")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.bar_chart(df["time"].value_counts())
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== MODEL INFO =====
    else:
        st.markdown("""
        <div class="card" style="animation:float 4s infinite, glow 2s infinite;">
            ✔ Supervised ML Classification<br><br>
            ✔ Scikit-learn Pipeline<br><br>
            ✔ Lunch vs Dinner Prediction<br><br>
        </div>
        """, unsafe_allow_html=True)

    # ===== FOOTER =====
    st.markdown("<div class='footer'><span>© 2026 Developed | by Prajwal Rajput</span></div>", unsafe_allow_html=True)

    # ===== TOAST =====
    if st.session_state.toast:
        st.markdown(f"<div class='toast'>{st.session_state.toast}</div>", unsafe_allow_html=True)
        time.sleep(2)
        st.session_state.toast = ""

# ================= ROUTER =================
if st.session_state.logged_in: app()
else: login_page()
