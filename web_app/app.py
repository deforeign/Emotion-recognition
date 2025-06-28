import streamlit as st
import soundfile as sf

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict_emotion


st.set_page_config(page_title="Voice Emotion Detector", layout="centered")
st.title("🎙️ Voice Emotion Detector")

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    emotion = predict_emotion("temp.wav")
    st.success(f"Predicted Emotion: **{emotion.upper()}**")

