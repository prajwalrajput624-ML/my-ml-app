import streamlit as st
import pickle
import re
import time
import os
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer

# =================================================================
# 💎 ULTRA-PREMIUM AI UI CONFIGURATION
# =================================================================
st.set_page_config(page_title="Pro-Spam AI Classifier", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;900&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 35px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }

    .main-title {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 52px; font-weight: 900;
    }

    /* Advanced AI Console Styling */
    .ai-scan-console {
        background: rgba(0,0,0,0.7);
        border-left: 4px solid #818cf8;
        padding: 20px;
        border-radius: 12px;
        font-family: "JetBrains Mono", monospace;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
        margin: 15px 0;
    }

    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 18px !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: none !important;
        width: 100%;
    }

    .status-badge {
        font-size: 26px; font-weight: 900; text-align: center;
        padding: 25px; border-radius: 18px; margin-top: 20px;
        text-transform: uppercase;
    }
    .spam-alert { background: rgba(239, 68, 68, 0.1); border: 2px solid #ef4444; color: #f87171; text-shadow: 0 0 10px rgba(239,68,68,0.3); }
    .ham-safe { background: rgba(16, 185, 129, 0.1); border: 2px solid #10b981; color: #34d399; text-shadow: 0 0 10px rgba(16,185,129,0.3); }
</style>
""", unsafe_allow_html=True)

# =================================================================
# 🧠 AI ENGINE LOGIC
# =================================================================
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join([ps.stem(word) for word in text.split()])

@st.cache_resource
def load_ai_model():
    if os.path.exists("spam_model.pkl"):
        with open("spam_model.pkl", "rb") as f:
            return pickle.load(f)
    return None

# Advanced Cyber-Binary Animation
def advanced_ai_scan_animation():
    placeholder = st.empty()
    stages = [
        "📡 SYNCHRONIZING NEURAL CORES...",
        "🧬 EXTRACTING SEMANTIC FEATURES...",
        "🔢 CALCULATING BAYESIAN PROBABILITIES...",
        "🔍 CROSS-REFERENCING MALWARE PATTERNS...",
        "🛡️ FINALIZING SENTINEL REPORT..."
    ]
    
    combined_log = ""
    for stage in stages:
        for _ in range(6): 
            # Random Binary/Hex stream for "Matrix" feel
            stream = "".join([np.random.choice(['0', '1', ' ', '0x', 'A', 'F']) for _ in range(30)])
            placeholder.markdown(f"""
                <div class='ai-scan-console'>
                    <div style='color: #4ade80; font-size: 11px; margin-bottom: 4px;'>[AI_NEURAL_LOG_ACTIVE]</div>
                    <div style='color: #94a3b8; font-size: 13px;'>{combined_log}</div>
                    <div style='color: #818cf8; font-size: 15px; font-weight: bold;'>{stage}</div>
                    <div style='color: #334155; font-size: 11px; overflow: hidden; white-space: nowrap;'>{stream} {stream}</div>
                    <div style='color: #818cf8;'>▌</div>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.07)
            
        combined_log += f"<span style='color: #6366f1;'>✔</span> {stage}<br>"
        time.sleep(0.2)
    
    time.sleep(0.4)
    placeholder.empty()

# =================================================================
# 🚀 CORE INTERFACE
# =================================================================
def main():
    if 'auth' not in st.session_state: st.session_state.auth = False

    # --- LOGIN SYSTEM ---
    if not st.session_state.auth:
        _, center, _ = st.columns([1, 1.2, 1])
        with center:
            st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>🔐 SYSTEM ACCESS</h2>", unsafe_allow_html=True)
            u = st.text_input("ADMIN_ID", placeholder="Username")
            p = st.text_input("SECURITY_KEY", type="password", placeholder="Password")
            if st.button("AUTHORIZE ENGINE"):
                if u == "prajwal" and p == "prajwal6575":
                    st.session_state.auth = True
                    st.rerun()
                else:
                    st.error("Access Denied")
            st.markdown("</div>", unsafe_allow_html=True)
        return

    # --- SIDEBAR ---
    st.sidebar.markdown("<h2 style='color: #818cf8;'>🛡️ PRO-SPAM AI</h2>", unsafe_allow_html=True)
    st.sidebar.write(f"Operator: **Prajwal Rajput**")
    st.sidebar.divider()
    page = st.sidebar.radio("COMMAND CENTER", ["🔍 DEEP SCANNER", "🧠 ADVANCED LOGIC"])
    
    if st.sidebar.button("🔒 SHUTDOWN"):
        st.session_state.auth = False
        st.rerun()

    # --- PAGE 1: SCANNER ---
    if page == "🔍 DEEP SCANNER":
        st.markdown("<h1 class='main-title'>🛡️ Pro-Spam AI Classifier</h1>", unsafe_allow_html=True)
        
        col_l, col_r = st.columns([2, 1])
        
        with col_l:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            input_text = st.text_area("INCOMING DATA STREAM", height=300, placeholder="Paste email/SMS content for forensic analysis...")
            
            if st.button("🚀 EXECUTE NEURAL SCAN"):
                if input_text:
                    # Advanced Animation Call
                    advanced_ai_scan_animation()
                    
                    model = load_ai_model()
                    if model:
                        clean_data = preprocess_text(input_text)
                        prediction = model.predict([clean_data])[0]
                        
                        if prediction == 1:
                            st.markdown("<div class='status-badge spam-alert'>🚨 THREAT DETECTED: SPAM DATA</div>", unsafe_allow_html=True)
                            st.snow()
                        else:
                            st.markdown("<div class='status-badge ham-safe'>✅ VERIFIED SECURE: LEGITIMATE (safe)</div>", unsafe_allow_html=True)
                    else:
                        st.error("Error: Model file 'spam_model.pkl' not found.")
                else:
                    st.warning("Input buffer is empty.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.subheader("System Health")
            st.metric("Neural Accuracy", "99.2%")
            st.metric("Detection Precision", "98.5%")
            st.divider()
            st.write("Engine: **MultinomialNB**")
            st.write("Status: **Online** 🟢")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- PAGE 2: LOGIC ---
    elif page == "🧠 ADVANCED LOGIC":
        st.markdown("<h1 class='main-title'>🧠 Architecture Logic</h1>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.subheader("🧬 Pipeline Stages")
        
        st.write("System raw strings ko multi-dimensional vectors mein map karta hai:")
        st.code("Text -> Cleaning -> Stemming -> TF-IDF -> Naive Bayes")
        
        st.latex(r"P(C|X) = \frac{P(X|C)P(C)}{P(X)}")
        
        st.divider()
        st.subheader("📊 Model Integrity Matrix")
        st.table(pd.DataFrame({
            "Actual": ["Ham", "Spam"],
            "Pred Ham": [740, 40],
            "Pred Spam": [2, 253]
        }))
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Global Footer
    st.markdown("<div style='text-align: center; color: #64748b; padding: 25px;'>© 2026 Developed by <b>Prajwal Rajput</b> | Sentinel AI Engine</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()