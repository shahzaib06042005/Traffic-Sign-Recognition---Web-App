import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH  = "traffic_sign_model.h5"
LABELS_PATH = "labels.txt"
IMG_HEIGHT  = 60
IMG_WIDTH   = 60

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sign emoji / colour mapping ──────────────────────────────────────────────
SIGN_META = {
    "No_Entry":      {"emoji": "🚫", "color": "#FF4757", "desc": "No vehicles may enter"},
    "Pedestrians":   {"emoji": "🚶", "color": "#2ED573", "desc": "Pedestrian crossing ahead"},
    "Priority_Road": {"emoji": "🔷", "color": "#1E90FF", "desc": "You have the right of way"},
    "Roundabout":    {"emoji": "🔄", "color": "#FFA502", "desc": "Enter the roundabout"},
    "Speed_Limit_30":{"emoji": "🔢", "color": "#FF6B81", "desc": "Maximum speed: 30 km/h"},
    "Speed_Limit_50":{"emoji": "🔢", "color": "#FF6348", "desc": "Maximum speed: 50 km/h"},
    "Stop":          {"emoji": "🛑", "color": "#FF4757", "desc": "Come to a complete stop"},
    "Turn_Left":     {"emoji": "⬅️", "color": "#7BED9F", "desc": "Turn left ahead"},
    "Turn_Right":    {"emoji": "➡️", "color": "#70A1FF", "desc": "Turn right ahead"},
    "Yield":         {"emoji": "⚠️", "color": "#ECCC68", "desc": "Yield to oncoming traffic"},
}

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Animated gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    color: #e0e0e0;
}
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

/* ── Sidebar glass ── */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.75) !important;
    backdrop-filter: blur(18px) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* ── Header banner ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.hero p {
    color: rgba(200,200,220,0.75);
    font-size: 1.1rem;
    font-weight: 300;
    margin: 0;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 20px;
    padding: 1.75rem 2rem;
    backdrop-filter: blur(14px);
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 40px rgba(0,0,0,0.5);
}

/* ── Prediction result card ── */
.result-card {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
@keyframes popIn {
    0%   { opacity: 0; transform: scale(0.85); }
    100% { opacity: 1; transform: scale(1);    }
}
.result-emoji { font-size: 5rem; line-height: 1.1; margin-bottom: 0.5rem; }
.result-label { font-size: 2rem; font-weight: 700; color: #fff; margin-bottom: 0.25rem; }
.result-desc  { font-size: 1rem; color: rgba(220,220,255,0.7); margin-bottom: 1rem; }

/* ── Confidence bar ── */
.conf-bar-wrap { width: 100%; background: rgba(255,255,255,0.1); border-radius: 100px; height: 14px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 100px; transition: width 0.8s ease; }

/* ── Top-3 chips ── */
.chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; justify-content: center; margin-top: 1rem; }
.chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 100px;
    padding: 0.3rem 0.85rem;
    font-size: 0.82rem;
    color: #c8c8e8;
}
.chip strong { color: #fff; }

/* ── Warning / No-sign card ── */
.warn-card {
    background: rgba(255, 180, 0, 0.10);
    border: 1px solid rgba(255, 180, 0, 0.30);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    color: #ffd166;
    font-size: 1.05rem;
    animation: popIn 0.4s ease;
}

/* ── Sidebar stat pill ── */
.stat-pill {
    background: rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ── File uploader / camera area ── */
[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 16px !important;
    color: #c8c8e8 !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Expander ── */
details { background: rgba(255,255,255,0.04) !important; border-radius: 12px !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #a78bfa !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = []
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        classes = ["Unknown"] * 100
    return model, classes

model, classes = load_model_and_labels()

# ─── Hero header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🚦 Traffic Sign Recognition</h1>
    <p>AI-powered sign detection — upload an image or capture with your camera</p>
</div>
""", unsafe_allow_html=True)

# ─── Error guard ──────────────────────────────────────────────────────────────
if model is None:
    st.markdown(f"""
    <div class="warn-card">
        ⚠️ &nbsp; Model file <code>{MODEL_PATH}</code> not found.<br>
        <small>Please run <code>python train.py</code> to train the model first.</small>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Prediction helper ────────────────────────────────────────────────────────
def predict_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)
    probs = preds[0]
    top3_idx  = np.argsort(probs)[-3:][::-1]
    top3      = [(classes[i], float(100 * probs[i])) for i in top3_idx]
    best_idx  = int(np.argmax(probs))
    conf      = float(100 * probs[best_idx])
    label     = classes[best_idx] if conf >= 40 else "No Sign Detected"
    return label, conf, top3

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio("Input source", ["📤 Upload Image", "📷 Webcam"], label_visibility="collapsed")
    st.markdown("---")

    st.markdown("### 🏷️ Recognisable Signs")
    for sign in classes:
        meta = SIGN_META.get(sign, {"emoji": "🔹", "color": "#aaa", "desc": ""})
        st.markdown(f"""
        <div class="stat-pill">
            {meta['emoji']} &nbsp; <strong>{sign.replace('_',' ')}</strong><br>
            <span style="color:rgba(200,200,220,0.55);font-size:0.78rem;">{meta['desc']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("ℹ️ Add more signs"):
        st.markdown("""
1. Add images to `data/train/<SignName>/`
2. Add validation images to `data/val/<SignName>/`
3. Run `python train.py`
4. Restart this app
        """)

# ─── Main columns ─────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

input_image = None

with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if "Upload" in mode:
        st.markdown("#### 📤 Upload an image")
        uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded:
            input_image = Image.open(uploaded)
            st.image(input_image, use_container_width=True, caption="")
    else:
        st.markdown("#### 📷 Capture with webcam")
        cam = st.camera_input("", label_visibility="collapsed")
        if cam:
            input_image = Image.open(cam)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── Results ──────────────────────────────────────────────────────────────────
with col_result:
    if input_image is None:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 4rem 2rem; color: rgba(200,200,240,0.4);">
            <div style="font-size:3.5rem">🔍</div>
            <p style="font-size:1rem; margin-top:0.75rem;">Results will appear here<br>after you provide an image</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analysing…"):
            try:
                label, confidence, top3 = predict_image(input_image)

                if label == "No Sign Detected":
                    best_guess, best_conf = top3[0]
                    st.markdown(f"""
                    <div class="warn-card">
                        <div style="font-size:2.5rem">🤔</div>
                        <div style="font-size:1.3rem; font-weight:600; margin:0.5rem 0;">No Clear Sign Detected</div>
                        <div style="font-size:0.9rem; opacity:0.8;">
                            Best guess: <strong>{best_guess.replace('_',' ')}</strong>
                            &nbsp;({best_conf:.1f}%)
                        </div>
                        <div style="font-size:0.82rem; margin-top:0.5rem; opacity:0.6;">
                            Confidence below 40%. Try a clearer image.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    meta  = SIGN_META.get(label, {"emoji": "🔹", "color": "#a78bfa", "desc": ""})
                    color = meta["color"]
                    bar_w = int(confidence)

                    # Top-3 chips HTML
                    chips_html = "".join(
                        f'<div class="chip"><strong>{nm.replace("_"," ")}</strong> &nbsp; {pr:.1f}%</div>'
                        for nm, pr in top3
                    )

                    st.markdown(f"""
                    <div class="result-card"
                         style="background: linear-gradient(135deg, {color}22, {color}08);
                                border: 1px solid {color}55;">
                        <div class="result-emoji">{meta['emoji']}</div>
                        <div class="result-label">{label.replace('_', ' ')}</div>
                        <div class="result-desc">{meta['desc']}</div>

                        <div style="margin: 1rem 0 0.35rem; font-size:0.82rem;
                                    color:rgba(220,220,255,0.55);">
                            Confidence
                        </div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill"
                                 style="width:{bar_w}%;
                                        background: linear-gradient(90deg, {color}99, {color});"></div>
                        </div>
                        <div style="font-size:1.4rem; font-weight:700;
                                    color:{color}; margin-top:0.5rem;">
                            {confidence:.1f}%
                        </div>

                        <div style="font-size:0.78rem; color:rgba(200,200,240,0.45);
                                    margin-top:1.25rem; margin-bottom:0.3rem;">
                            TOP 3 PREDICTIONS
                        </div>
                        <div class="chip-row">{chips_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="warn-card" style="border-color: rgba(255,80,80,0.4); color:#ff8080;">
                    ❌ &nbsp; Prediction failed: <code>{e}</code>
                </div>
                """, unsafe_allow_html=True)
