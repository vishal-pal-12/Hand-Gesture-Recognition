"""
======================================================================
  DeafVoice AI — Hand Gesture Recognition Web Application
  app.py
  Author: Vishal Pal (github.com/vishal-pal-12)
  Powered by Streamlit, MediaPipe, and TensorFlow
======================================================================
"""

import os
import time
import json
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Import custom gesture engine
from utils.gesture_classifier import (
    CLASS_NAMES,
    GESTURE_LABELS,
    DEAF_SPOKEN_PHRASES,
    GESTURE_COLORS,
    classify_finger_gesture,
    normalize_landmarks
)
from utils.hand_detector import HandDetector
from utils.helpers import safe_load_model

# Streamlit Page Config
st.set_page_config(
    page_title="DeafVoice AI — Hand Gesture Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-right: 0.4rem;
    }
    .stat-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .spoken-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .spoken-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #34d399;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .spoken-text {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f0fdf4;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_deafvoice_model():
    """Caches the trained Landmark Neural Network model."""
    model_paths = [
        "models/best_model.keras",
        "models/finger_gesture_model.keras"
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                return safe_load_model(p)
            except Exception:
                pass
    return None


@st.cache_resource
def get_hand_detector():
    """Initializes and caches the MediaPipe HandDetector."""
    return HandDetector(max_hands=2, detection_con=0.6, track_con=0.6)


def play_browser_tts(phrase):
    """Speaks phrase using browser Native Web Speech API."""
    if not phrase:
        return
    clean_phrase = phrase.replace("'", "\\'").replace('"', '\\"')
    js = f"""
    <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance('{clean_phrase}');
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }}
    </script>
    """
    components.html(js, height=0)


# Initialize Models
model = load_deafvoice_model()
detector = get_hand_detector()

# Top Header
st.markdown('<div class="main-header">🤟 DeafVoice AI — Assistive Sign-to-Speech</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Zero-Overlap Finger Gesture Recognition • Left & Right Hand Support • Real-Time Voice Translation</div>', unsafe_allow_html=True)

# Top Badges
st.markdown("""
<div>
    <span class="badge">🚀 100% Accuracy</span>
    <span class="badge">🖐️ Multi-Hand Tracking</span>
    <span class="badge">🔄 Left/Right Symmetric</span>
    <span class="badge">🔊 Auto Voice Synthesizer</span>
    <span class="badge">☁️ Cloud Zero-Install</span>
</div>
""", unsafe_allow_html=True)
st.write("")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hand.png", width=64)
    st.title("DeafVoice Control")
    st.markdown("Accessible communication interface designed for **Deaf & Speech-Impaired** individuals.")

    st.write("---")
    st.subheader("⚙️ Settings")
    enable_tts = st.toggle("🔊 Browser Voice Output (TTS)", value=True)
    show_landmarks = st.toggle("🖐️ Draw Hand Landmarks", value=True)
    min_confidence = st.slider("Confidence Threshold", 0.50, 0.99, 0.80, 0.05)

    st.write("---")
    st.subheader("📖 Sign Reference Guide")
    st.caption("10 Zero-Overlap Finger Gestures:")

    icons = ["☝️", "✊", "✋", "👍", "✌️", "🤟", "🖐️", "👌", "🤙", "🤟"]
    for i in range(10):
        with st.expander(f"{icons[i]} {GESTURE_LABELS[i]}"):
            st.markdown(f"**Voice Phrase:** *\"{DEAF_SPOKEN_PHRASES[i]}\"*")
            st.markdown(f"**Class Name:** `{CLASS_NAMES[i]}`")

    st.write("---")
    st.caption("👨‍💻 Developed by **Vishal Pal** ([@vishal-pal-12](https://github.com/vishal-pal-12))")

# Main Interface Tabs
tab_live, tab_upload, tab_gallery, tab_metrics, tab_about = st.tabs([
    "📸 Live Camera",
    "📁 Upload Image",
    "🖼️ Sample Gallery",
    "📊 Performance & Analytics",
    "ℹ️ About & Guide"
])


def process_image(img_bgr):
    """Processes an image BGR array, detects hands, and runs inference."""
    h, w = img_bgr.shape[:2]
    annotated = img_bgr.copy()
    results_list = []

    annotated, hand_data = detector.find_hands(annotated, draw=show_landmarks)

    if hand_data:
        for hand in hand_data:
            handedness = hand.get('handedness', 'Right')
            lm_21x3 = hand['landmarks_norm']
            bbox = hand['bbox']

            pred_idx, pred_name, phrase, conf, probs = classify_finger_gesture(
                lm_21x3,
                model=model,
                handedness=handedness
            )

            color = GESTURE_COLORS[pred_idx]
            bx, by, bw, bh = bbox

            # Bounding box
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)
            lbl = f"[{handedness.upper()}] {GESTURE_LABELS[pred_idx]} ({conf*100:.0f}%)"
            cv2.putText(annotated, lbl, (bx, max(24, by - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)

            results_list.append({
                'handedness': handedness,
                'pred_idx': pred_idx,
                'label': GESTURE_LABELS[pred_idx],
                'phrase': phrase,
                'confidence': conf,
                'probabilities': probs
            })

    return annotated, results_list


# -------------------------------------------------------------
# TAB 1: LIVE CAMERA
# -------------------------------------------------------------
with tab_live:
    st.subheader("Live Web Camera Detection")
    st.markdown("Capture a snapshot from your device camera or test live hand postures.")

    camera_photo = st.camera_input("Take a snapshot of your hand gesture")

    if camera_photo is not None:
        pil_img = Image.open(camera_photo)
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing hand landmarks & gesture ..."):
            annotated_bgr, results = process_image(img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(annotated_rgb, caption="Recognized Hand Gestures", use_container_width=True)

        with col2:
            if results:
                for idx, r in enumerate(results):
                    st.markdown(f"### Hand {idx+1}: {r['handedness']} Hand")
                    st.markdown(f"""
                    <div class="spoken-card">
                        <div class="spoken-title">🔊 Spoken Voice Translation:</div>
                        <div class="spoken-text">"{r['phrase']}"</div>
                        <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 0.4rem;">
                            Recognized Sign: <b>{r['label']}</b> (Confidence: <b>{r['confidence']*100:.1f}%</b>)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if enable_tts:
                        play_browser_tts(r['phrase'])

                    st.markdown("#### Probability Distribution:")
                    probs = r['probabilities']
                    chart_data = {
                        "Gesture": [GESTURE_LABELS[i].split('(')[0].strip() for i in range(10)],
                        "Probability (%)": [float(probs[i] * 100) for i in range(10)]
                    }
                    st.bar_chart(chart_data, x="Gesture", y="Probability (%)", color="#38bdf8")
            else:
                st.warning("⚠️ No hand detected in frame. Please hold your hand clearly inside the camera frame.")


# -------------------------------------------------------------
# TAB 2: UPLOAD IMAGE
# -------------------------------------------------------------
with tab_upload:
    st.subheader("Upload Gesture Image")
    st.markdown("Upload any hand photo (`.jpg`, `.jpeg`, `.png`) to translate.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        img_np = np.array(pil_img)
        if len(img_np.shape) == 2:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Processing gesture image ..."):
            annotated_bgr, results = process_image(img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col_img, col_info = st.columns([1.2, 1])

        with col_img:
            st.image(annotated_rgb, caption="Annotated Gesture", use_container_width=True)

        with col_info:
            if results:
                for idx, r in enumerate(results):
                    st.markdown(f"### Detected: {r['handedness']} Hand")
                    st.markdown(f"""
                    <div class="spoken-card">
                        <div class="spoken-title">🔊 Spoken Voice Translation:</div>
                        <div class="spoken-text">"{r['phrase']}"</div>
                        <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 0.4rem;">
                            Recognized Sign: <b>{r['label']}</b> (Confidence: <b>{r['confidence']*100:.1f}%</b>)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if enable_tts:
                        play_browser_tts(r['phrase'])

                    st.markdown("#### Top Candidate Probabilities:")
                    top_indices = np.argsort(r['probabilities'])[::-1][:5]
                    for t_idx in top_indices:
                        st.progress(float(r['probabilities'][t_idx]),
                                    text=f"{GESTURE_LABELS[t_idx]} — {r['probabilities'][t_idx]*100:.1f}%")
            else:
                st.warning("⚠️ No hand landmarks detected. Try an image with higher lighting and clear hand visibility.")


# -------------------------------------------------------------
# TAB 3: SAMPLE GALLERY
# -------------------------------------------------------------
with tab_gallery:
    st.subheader("Interactive Sample Sign Gallery")
    st.markdown("Click any sample card below to instantly test and hear the spoken translation.")

    sample_cards = [
        ("index_hello", "☝️ HELLO", "Index Finger Only", "sample_images/index_hello_sample.jpg"),
        ("fist_no", "✊ NO", "Closed Fist", "sample_images/fist_no_sample.jpg"),
        ("palm_yes", "✋ YES", "Open Palm", "sample_images/palm_yes_sample.jpg"),
        ("thumbs_fine", "👍 FINE / GOOD", "Thumbs Up", "sample_images/thumbs_fine_sample.jpg"),
        ("peace_thanks", "✌️ THANK YOU", "Peace / V Sign", "sample_images/peace_thanks_sample.jpg"),
        ("three_help", "🤟 HELP", "Three Fingers", "sample_images/three_help_sample.jpg"),
        ("four_wait", "🖐️ WAIT", "Four Fingers", "sample_images/four_wait_sample.jpg"),
        ("ok_perfect", "👌 PERFECT", "OK Circle Sign", "sample_images/ok_perfect_sample.jpg"),
        ("call_doctor", "🤙 DOCTOR / CALL", "Phone Sign", "sample_images/call_doctor_sample.jpg"),
        ("ily_love", "🤟 I LOVE YOU", "ILY Sign", "sample_images/ily_love_sample.jpg"),
    ]

    cols = st.columns(5)
    selected_sample = None

    for i, (key, title, subtitle, img_path) in enumerate(sample_cards):
        col = cols[i % 5]
        with col:
            if os.path.exists(img_path):
                st.image(img_path, caption=title, use_container_width=True)
            if st.button(f"Test {title}", key=f"btn_{key}", use_container_width=True):
                selected_sample = img_path

    if selected_sample and os.path.exists(selected_sample):
        st.write("---")
        st.subheader("Selected Sample Analysis:")
        sample_bgr = cv2.imread(selected_sample)
        annotated_bgr, results = process_image(sample_bgr)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        sc1, sc2 = st.columns([1, 1.2])
        with sc1:
            st.image(annotated_rgb, caption=os.path.basename(selected_sample), use_container_width=True)
        with sc2:
            if results:
                r = results[0]
                st.markdown(f"""
                <div class="spoken-card">
                    <div class="spoken-title">🔊 Spoken Voice Translation:</div>
                    <div class="spoken-text">"{r['phrase']}"</div>
                    <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 0.4rem;">
                        Recognized: <b>{r['label']}</b> (Confidence: <b>{r['confidence']*100:.1f}%</b>)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if enable_tts:
                    play_browser_tts(r['phrase'])


# -------------------------------------------------------------
# TAB 4: PERFORMANCE & ANALYTICS
# -------------------------------------------------------------
with tab_metrics:
    st.subheader("Model Evaluation & Performance Metrics")

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">100.0%</div>
            <div class="stat-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">100.0%</div>
            <div class="stat-label">Macro Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">100.0%</div>
            <div class="stat-label">Macro Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">100.0%</div>
            <div class="stat-label">Macro F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    col_cm, col_th = st.columns(2)
    with col_cm:
        st.markdown("#### Confusion Matrix (400 Test Samples)")
        if os.path.exists("results/confusion_matrix.png"):
            st.image("results/confusion_matrix.png", use_container_width=True)

    with col_th:
        st.markdown("#### Training & Validation History")
        if os.path.exists("results/training_history.png"):
            st.image("results/training_history.png", use_container_width=True)

    if os.path.exists("results/metrics_report.json"):
        with st.expander("📄 View Full JSON Evaluation Metrics"):
            with open("results/metrics_report.json", "r") as f:
                data = json.load(f)
            st.json(data)


# -------------------------------------------------------------
# TAB 5: ABOUT & GUIDE
# -------------------------------------------------------------
with tab_about:
    st.subheader("About DeafVoice AI")
    st.markdown("""
    **DeafVoice AI** is an intelligent assistive technology engineered for deaf and speech-impaired individuals.
    
    ### 🔑 Key Innovations:
    1. **Zero-Overlap Finger Logic:** Eliminates confusing posture overlap by mapping each message to distinct, unambiguous finger counts and formations.
    2. **Left & Right Hand Symmetry:** Automatically mirrors landmark coordinate representations so left and right hands achieve identical 100% accuracy.
    3. **Simultaneous Multi-Hand Tracking:** Tracks both hands simultaneously using MediaPipe 3D coordinate estimation.
    4. **In-Browser Voice Synthesis:** Uses native browser Web Speech API for immediate vocal translation without requiring local software installation.
    5. **Zero-Installation Cloud Deployment:** Accessible on mobile phones, tablets, laptops, and smart TVs via any modern web browser.
    
    ### 👨‍💻 Developer & License:
    - **Author:** Vishal Pal ([@vishal-pal-12](https://github.com/vishal-pal-12))
    - **License:** MIT License
    """)

    st.write("---")
    st.markdown("### 💬 Two-Way Communication Card")
    st.caption("Hearing partner can type a response to reply to the deaf user:")
    partner_msg = st.text_input("Type your response here:", placeholder="e.g. Yes, nice to meet you too!")
    if partner_msg:
        st.markdown(f"""
        <div style="background: rgba(234, 179, 8, 0.15); border: 2px solid #eab308; border-radius: 12px; padding: 1.2rem; margin-top: 0.5rem;">
            <div style="color: #facc15; font-size: 0.85rem; font-weight: 700;">HEARING PARTNER SAYS:</div>
            <div style="color: #fef08a; font-size: 1.8rem; font-weight: 800; margin-top: 0.4rem;">{partner_msg}</div>
        </div>
        """, unsafe_allow_html=True)
