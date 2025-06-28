import streamlit as st
from st_audiorec import st_audiorec
import tempfile
import os
import sys
import soundfile as sf

# Ensure the src/ folder is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_emotion

st.set_page_config(page_title="🎙️ Voice Emotion Detector", layout="centered")
st.title("🎤 Voice Emotion Detector")

st.markdown("Choose an option below to analyze your emotion from speech:")

# User selects input method
option = st.radio("Input Method", ["🎙️ Record from Mic", "📁 Upload Audio File"], horizontal=True)

audio_path = None
audio_bytes = None

if option == "🎙️ Record from Mic":
    st.info("Click the mic button and speak for 2–4 seconds.")
    audio_bytes = st_audiorec()
    if audio_bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(audio_bytes)
            audio_path = tmpfile.name
        st.audio(audio_bytes, format="audio/wav")

elif option == "📁 Upload Audio File":
    uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            audio_path = tmpfile.name
        st.audio(uploaded_file, format="audio/wav")

# If we have an audio path, try predicting
if audio_path:
    try:
        emotion = predict_emotion(audio_path)
        st.success(f"🧠 Predicted Emotion: **{emotion.upper()}**")
    except Exception as e:
        st.error("❌ Failed to predict emotion.")
        st.exception(e)
