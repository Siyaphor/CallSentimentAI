import streamlit as st
import whisper
from transformers import pipeline
from pydub import AudioSegment
import os

st.title(" Call Sentiment Analyzer")

# Load models
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    sentiment = pipeline("sentiment-analysis")
    return whisper_model, sentiment

whisper_model, sentiment_model = load_models()

uploaded_file = st.file_uploader("Upload audio", type=["mp3","wav","mp4","m4a"])

if uploaded_file is not None:
    st.write("Processing audio...")

    # Save temp file
    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.read())

    # Convert to wav
    audio = AudioSegment.from_file("temp_audio")
    audio.export("converted.wav", format="wav")

    # Transcribe
    result = whisper_model.transcribe("converted.wav")
    text = result["text"]

    st.subheader("Transcript")
    st.write(text)

    # Sentiment
    sentiment = sentiment_model(text)[0]

    st.subheader("Sentiment")
    st.write(sentiment["label"])
    st.write(f"Confidence: {sentiment['score']:.2f}")