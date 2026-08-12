"""
Streamlit frontend for Shoulder X-ray AI Analysis Platform.
v3.0 - PLATINUM EDITION (Cyber-Medical UI + PDF Reporting).
"""

import io
import base64
import requests
import time
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SHOULDER.AI PLATINUM | Neural Diagnostics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────

if 'history' not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# Backend URL
# ─────────────────────────────────────────────

BACKEND_URL = st.sidebar.text_input(
    "🔗 Backend API URL",
    value="http://localhost:8000",
)

# ─────────────────────────────────────────────
# Aurora Clinical CSS — Premium Medical UI
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;600;700&display=swap');

    :root {
        --primary-emerald: #10b981;
        --accent-indigo: #6366f1;
        --error-rose: #f43f5e;
        --bg-deep: #0b0e14;
        --surface-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.08);
        --text-muted: #94a3b8;
    }

    .stApp {
        background: radial-gradient(circle at 50% 50%, #111827 0%, #0b1120 100%);
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: rgba(11, 17, 32, 0.95) !important;
        backdrop-filter: blur(25px);
        border-right: 1px solid var(--border-glass);
    }

    /* ── Aurora Hero ── */
    .aurora-hero {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.03), transparent);
        padding: 4rem 2rem;
        border-radius: 40px;
        text-align: center;
        margin-bottom: 3.5rem;
        border: 1px solid var(--border-glass);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .aurora-hero::after {
        content: "";
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 30%, rgba(16, 185, 129, 0.05) 0%, transparent 70%);
        pointer-events: none;
    }

    .aurora-hero h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        letter-spacing: -1.5px;
        background: linear-gradient(to right, #10b981, #34d399, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .aurora-hero .tagline {
        font-size: 1.1rem;
        color: var(--primary-emerald);
        font-weight: 600;
        letter-spacing: 5px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .aurora-hero .version-badge {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    /* ── Diagnostic Cards ── */
    .diag-card {
        background: var(--surface-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: 28px;
        padding: 2rem;
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .diag-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
        border-color: rgba(16, 185, 129, 0.2);
    }

    /* ── Scanning Overlay ── */
    .scanline {
        width: 100%;
        height: 2px;
        background: linear-gradient(to right, transparent, var(--primary-emerald), transparent);
        position: absolute;
        top: 0; left: 0;
        z-index: 10;
        animation: scan 4s ease-in-out infinite;
        opacity: 0.6;
        box-shadow: 0 0 15px var(--primary-emerald);
    }
    @keyframes scan {
        0% { top: 0%; opacity: 0; }
        10% { opacity: 0.8; }
        90% { opacity: 0.8; }
        100% { top: 100%; opacity: 0; }
    }

    /* ── Badges & Status ── */
    .res-badge {
        padding: 0.8rem 2.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.6rem;
        text-align: center;
        letter-spacing: 1px;
        display: inline-block;
        margin-top: 1rem;
    }
    .res-normal { border: 1px solid #10b981; color: #10b981; background: rgba(16, 185, 129, 0.05); }
    .res-abnormal { border: 1px solid #f43f5e; color: #f43f5e; background: rgba(244, 63, 94, 0.05); }

    /* ── Premium Buttons ── */
    .stButton>button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white !important;
        border: none;
        padding: 1.2rem 2.5rem;
        border-radius: 14px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
    }

    /* ── Clinical Report Logic View ── */
    .report-box {
        background: #020617;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.9rem;
        color: #f1f5f9;
        line-height: 1.7;
        border-left: 8px solid var(--accent-indigo);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — Session Archives
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎞️ DIAGNOSTIC ARCHIVE")
    if not st.session_state.history:
        st.info("No active records in this session.")
    else:
        for idx, entry in enumerate(reversed(st.session_state.history)):
            border_color = "#10b981" if entry['prediction'] == "NORMAL" else "#f43f5e"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; margin-bottom: 12px; border-left: 4px solid {border_color};">
                <small style="color: #64748b;">CASE: #{len(st.session_state.history)-idx} — {entry['time']}</small><br>
                <div style="font-weight: 600; font-size: 0.9rem;">{entry['prediction']}</div>
                <div style="color: #475569; font-size: 0.8rem;">Confidence: {entry['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero Section — Aurora Evolution
# ─────────────────────────────────────────────

st.markdown("""
<div class="aurora-hero">
    <div class="tagline">Clinical Neural Processor</div>
    <h1>SHOULDER.AI <span class="version-badge">AURORA v4.1</span></h1>
    <p style="color: #94a3b8; font-size: 1.1rem;">High-fidelity radiographic diagnostic engine with automated feature mapping.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Core Interface — Diagnostic Center
# ─────────────────────────────────────────────

c1, c2 = st.columns([1.2, 0.8])

with c1:
    st.markdown('<div class="diag-card">', unsafe_allow_html=True)
    st.markdown("### 📥 RADIOGRAPH INTAKE")
    uploaded_file = st.file_uploader("Upload DICOM/Standard X-ray Scan (JPEG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Scanline effect wrapper
        st.markdown('<div style="position: relative;">', unsafe_allow_html=True)
        st.image(image, caption="Radiograph Input Buffer", use_container_width=True)
        # Visual decorator
        st.markdown('<div class="scanline"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="diag-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ SYSTEM PARAMETERS")
    st.markdown(f"""
    <div style="font-size: 0.9rem; color: #94a3b8;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span>Architecture</span> <span style="color: #6366f1;">EfficientNet-B0</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span>Resolution</span> <span style="color: #6366f1;">224x224 (Resized)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span>Mapping</span> <span style="color: #6366f1;">Grad-CAM Heatmap</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Reporting</span> <span style="color: #6366f1;">PDF Aurora v4</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    if uploaded_file:
        analyze_btn = st.button("RUN CLINICAL DIAGNOSTIC", use_container_width=True)
    else:
        st.markdown('<p style="color: #475569; font-style: italic; text-align: center;">Waiting for radiograph upload...</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Analysis Logic — Clinical Processing
# ─────────────────────────────────────────────

if uploaded_file and analyze_btn:
    # ── Custom Clinical Loading ──
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 2rem; margin-bottom: 1rem; color: #10b981; font-weight: 600;">EXECUTING NEURAL SCAN...</div>
            <div style="width: 100%; height: 6px; background: rgba(16, 185, 129, 0.1); border-radius: 10px; overflow: hidden; margin: 0 auto; max-width: 400px;">
                <div style="width: 30%; height: 100%; background: #10b981; animation: slide 1.5s infinite ease-in-out;"></div>
            </div>
            <p style="color: #64748b; margin-top: 1.5rem; font-size: 0.95rem;">Parsing anatomical structures & mapping feature weights...</p>
        </div>
        <style>
            @keyframes slide { 
                0% { transform: translateX(-100%); } 
                100% { transform: translateX(330%); } 
            }
        </style>
        """, unsafe_allow_html=True)

    try:
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        time.sleep(1.2) # Psychological "processing" time
        response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=60)
        
        loading_placeholder.empty()

        if response.status_code == 200:
            result = response.json()
            
            # Record in history
            st.session_state.history.append({
                "time": time.strftime("%H:%M:%S"),
                "prediction": result["prediction"],
                "confidence": result["confidence"]
            })

            # ─────────────────────────────────────────────
            # Result Display — Primary Diagnostic
            # ─────────────────────────────────────────────
            st.markdown('<div class="diag-card" style="border-left: 10px solid #10b981;">' if result["prediction"] == "NORMAL" else '<div class="diag-card" style="border-left: 10px solid #f43f5e;">', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown("### 🏆 MAIN DIAGNOSTIC")
                badge_style = "res-normal" if result["prediction"] == "NORMAL" else "res-abnormal"
                st.markdown(f'<div class="res-badge {badge_style}">{result["prediction"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<p style="color: #94a3b8; font-size: 0.85rem; margin-top: 10px;">Primary AI classification based on clinical image features.</p>', unsafe_allow_html=True)
                
            with res_col2:
                st.markdown("### 📈 CLASSIFICATION CONFIDENCE")
                conf_pct = result["confidence"] * 100
                st.markdown(f"<h1 style='margin-bottom: 0;'>{conf_pct:.2f}%</h1>", unsafe_allow_html=True)
                st.progress(result["confidence"])
                st.markdown(f'<p style="color: #94a3b8; font-size: 0.85rem;">Probability index of the final determination.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ─────────────────────────────────────────────
            # Feature Mapping — Main Cause Detection
            # ─────────────────────────────────────────────
            st.markdown("### 💠 FEATURE ACTIVATION MAPPING (THE 'WHY')")
            v1, v2 = st.columns([1, 1])
            
            with v1:
                st.markdown('<div class="diag-card">', unsafe_allow_html=True)
                st.markdown("#### ATTENTION HOTSPOTS")
                heatmap_bytes = base64.b64decode(result["heatmap"])
                st.image(Image.open(io.BytesIO(heatmap_bytes)), caption="Neural Attention Map (Grad-CAM Overlay)", use_container_width=True)
                st.markdown('<p style="color: #64748b; font-size: 0.8rem;">Brighter areas indicate regions that most influenced the AI diagnostic decision.</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with v2:
                st.markdown('<div class="diag-card">', unsafe_allow_html=True)
                st.markdown("#### CLINICAL REPORT (LOGS)")
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.text(result["report"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # PDF Download Button
                pdf_bytes = base64.b64decode(result["pdf"])
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📄 DOWNLOAD OFFICIAL CLINICAL REPORT (PDF)",
                    data=pdf_bytes,
                    file_name=f"Aurora_Report_{int(time.time())}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error(f"SYSTEM ERROR: {response.json().get('detail')}")
            
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"CONNECTION FAILURE: {str(e)}")

# ─────────────────────────────────────────────
# Footer — Clinical Compliance
# ─────────────────────────────────────────────

st.markdown("""
<div style="text-align: center; color: #475569; font-size: 0.85rem; padding: 5rem 0 2rem 0; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 5rem;">
    <div style="color: #94a3b8; font-weight: 600; margin-bottom: 5px;">SHOULDER.AI AURORA SYSTEMS</div>
    NOT FOR PRIMARY CLINICAL DIAGNOSIS | VERIFY WITH BOARD-CERTIFIED RADIOLOGIST<br>
    © 2026 NEURAL DIAGNOSTICS LABS. ALL TENSORS RESERVED.
</div>
""", unsafe_allow_html=True)
