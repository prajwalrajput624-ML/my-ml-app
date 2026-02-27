import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# ==========================================
# 1. PAGE CONFIG & DARK GLOW UI
# ==========================================
st.set_page_config(
    page_title="SkyCast AI Pro",
    page_icon="🌌",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #05070a; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #0a0f16; border-right: 1px solid #1e293b; }
    
    @keyframes moveGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .gemini-loader {
        height: 4px; width: 100%;
        background: linear-gradient(-45deg, #4285f4, #9b72cb, #d96570, #10b981);
        background-size: 400% 400%;
        animation: moveGradient 3s linear infinite;
        border-radius: 10px; margin: 20px 0;
        box-shadow: 0 0 15px rgba(155, 114, 203, 0.4);
    }

    .result-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 25px; border-radius: 20px;
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stButton>button {
        background: linear-gradient(90deg, #1e40af, #7c3aed);
        color: white; border: none; border-radius: 12px;
        font-weight: bold; padding: 0.8rem; transition: 0.4s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.5);
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION & LOGIN
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_page():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<br><br><h1 style='text-align: center; color: #818cf8;'>🔐SKYCAST-AI</h1>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if username == "prajwal" and password == "prajwal3565":
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Credentials")

# ==========================================
# 3. MAIN DASHBOARD LOGIC
# ==========================================
if not st.session_state['logged_in']:
    login_page()
else:
    with st.sidebar:
        st.markdown("<div style='text-align: center; padding: 10px;'><h2 style='color: #818cf8; margin-bottom: 0;'>SkyCast Admin</h2><p style='color: #94a3b8; font-size: 0.9em;'>Status: Active Session</p></div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("### 🧠 Model Intelligence")
        st.markdown("<div style='background: rgba(30, 41, 59, 0.7); padding: 15px; border-radius: 12px; border: 1px solid #1e293b;'><p style='color: #94a3b8; margin-bottom: 5px; font-size: 0.85em;'>Architecture</p><h4 style='color: #34d399; margin-top: 0;'>XGBoost Classifier</h4><hr style='opacity: 0.1; margin: 10px 0;'><div style='display: flex; justify-content: space-between;'><div><p style='color: #94a3b8; margin: 0; font-size: 0.75em;'>ROC AUC</p><p style='color: #818cf8; font-weight: bold; margin: 0;'>0.96+</p></div><div><p style='color: #94a3b8; margin: 0; font-size: 0.75em;'>ACCURACY</p><p style='color: #818cf8; font-weight: bold; margin: 0;'>92%</p></div></div></div>", unsafe_allow_html=True)
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- UPDATED HEADER WITH ICON ---
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 20px;'>
            <span style='font-size: 3.5rem;'>🌌</span>
            <div>
                <h1 style='color: #f8fafc; margin-bottom:0;'>SkyCast Weather Intelligence</h1>
                <p style='color: #94a3b8; margin-top:0;'>Deployed by <b>Prajwal Rajput</b> | 33-Dimensional Inference Engine</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    @st.cache_resource
    def load_model():
        try:
            return joblib.load("weather_model.joblib")
        except:
            return None

    model = load_model()

    st.subheader("📡 Environmental Sensors")
    c1, c2 = st.columns(2)
    with c1:
        country = st.selectbox("Region", ["India"])
        city = st.selectbox("Monitoring Node", ["Surat", "Ahmedabad", "Mumbai", "Delhi"])
        temp = st.slider("Temperature (°C)", -10.0, 50.0, 23.0)
        hum = st.slider("Humidity (%)", 0, 100, 74)
        wind_dir = st.selectbox("Wind Vector", ["SW", "N", "NE", "E", "SE", "S", "W", "NW"])

    with c2:
        press = st.slider("Barometric Pressure (mb)", 950, 1050, 1009)
        cloud = st.slider("Cloud Cover (%)", 0, 100, 25)
        wind_s = st.number_input("Velocity (kph)", 0.0, 150.0, 11.2)
        uv = st.slider("UV Radiation Index", 0.0, 15.0, 0.0)
        feels = st.slider("Apparent Temp (°C)", -10.0, 55.0, 25.0)

    if st.button("🚀 Run AI Forecasting Engine"):
        if model is None:
            st.error("Fatal Error: Model file 'weather_model.joblib' not found.")
        else:
            st.markdown('<div class="gemini-loader"></div>', unsafe_allow_html=True)
            status_placeholder = st.empty()
            for step in ["Initializing Neural Weights...", "Injecting 33 Features...", "Running XGBoost Inference..."]:
                status_placeholder.markdown(f"<p style='color:#818cf8; text-align:center;'>{step}</p>", unsafe_allow_html=True)
                time.sleep(0.5)
            
            # Dynamic Feature Logic
            derived_condition = "rain" if (hum > 80 or press < 1005) else "clear"
            precip_val = 2.5 if hum > 80 else 0.0
            vis_val = 5.0 if hum > 80 else 10.0

            data_dict = {
                "country": country.lower(), "location_name": city.lower(), "temperature_celsius": temp,
                "feels_like_celsius": feels, "humidity": hum, "pressure_mb": press, 
                "cloud": cloud, "wind_kph": wind_s, "wind_direction": wind_dir.lower(), "uv_index": uv,
                "timezone": "Asia/Kolkata", "moon_phase": "waxing", 
                "condition_text": derived_condition,
                "latitude": 21.17, "longitude": 72.83, "temperature_fahrenheit": temp * 1.8 + 32,
                "feels_like_fahrenheit": feels * 1.8 + 32, "wind_mph": wind_s * 0.62,
                "wind_degree": 180, "pressure_in": press * 0.029, 
                "precip_mm": precip_val,
                "precip_in": precip_val * 0.039,
                "visibility_km": vis_val, "visibility_miles": vis_val * 0.62, 
                "gust_kph": wind_s * 1.1, "gust_mph": wind_s * 0.7, 
                "air_quality_Carbon_Monoxide": 200.0, "air_quality_Sulphur_dioxide": 5.0, 
                "air_quality_PM2.5": 15.0, "air_quality_PM10": 25.0,
                "air_quality_us-epa-index": 1, "air_quality_gb-defra-index": 1,
                "moon_illumination": 45, "last_updated_epoch": int(time.time())
            }
            
            input_df = pd.DataFrame([data_dict])
            prob = float(model.predict_proba(input_df)[0][1])
            status_placeholder.empty()

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("<h3 style='color:#94a3b8; margin-top:0;'>Precipitation Risk</h3>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='color:#818cf8; font-size: 3.5em; margin:0;'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
                st.progress(prob)
            
            with r2:
                if prob >= 0.35:
                    st.markdown("<h2 style='color:#f87171; margin-top:0;'>🌧️ Rain Likely</h2>", unsafe_allow_html=True)
                    st.warning(f"**AI Tip:** Don't forget your umbrella ☔. Humidity is at {hum}%, it might feel a bit muggy outside.")
                else:
                    st.markdown("<h2 style='color:#34d399; margin-top:0;'>☀️ Clear Weather</h2>", unsafe_allow_html=True)
                    if uv > 6:
                        st.info(f"**AI Tip:** It's bright out! UV Index is {uv}, so wear sunglasses 🕶️.")
                    elif temp > 35:
                        st.info("**AI Tip:** It's getting hot! Stay hydrated 💧.")
                    else:
                        st.success("**AI Tip:** Perfect weather! 🌈 Great for outdoor activities.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; margin-top:80px; padding: 20px; color:#475569; border-top: 1px solid #1e293b;'><p style='margin:0; font-weight:bold; color:#818cf8;'>@2026 Deployed by | Prajwal Rajput</p></div>", unsafe_allow_html=True)
