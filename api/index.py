import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, request


APP_TITLE = "VoiceIQ"
APP_SUBTITLE = "Call Sentiment Intelligence"
HISTORY_PATH = Path(__file__).resolve().parents[1] / "call_history.json"

app = Flask(__name__)


def load_history() -> List[Dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def summarize_history(history: List[Dict]) -> Dict:
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    total = len(history)
    for item in history:
        label = item.get("sentiment", "NEUTRAL").upper()
        if label in counts:
            counts[label] += 1
    pos_pct = round(counts["POSITIVE"] / total * 100) if total else 0
    return {
        "total": total,
        "positive": counts["POSITIVE"],
        "negative": counts["NEGATIVE"],
        "neutral": counts["NEUTRAL"],
        "pos_pct": pos_pct,
    }


def load_sentiment_model():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Sentiment analysis dependencies are not installed in the Vercel runtime."
        ) from exc

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def transcribe_audio(file_storage) -> str:
    try:
        import whisper
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "Audio transcription dependencies are not installed in the Vercel runtime."
        ) from exc

    whisper_model = whisper.load_model("tiny")
    suffix = Path(file_storage.filename or "audio").suffix or ".wav"

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        wav_path = Path(temp_dir) / "converted.wav"
        file_storage.save(input_path)

        audio = AudioSegment.from_file(input_path)
        audio.export(wav_path, format="wav")

        result = whisper_model.transcribe(str(wav_path))
        return result.get("text", "")


@app.get("/")
def home():
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>VoiceIQ</title>
        <style>
          body {
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            background: #050810;
            color: #e2e8f0;
            font-family: Arial, sans-serif;
          }
          main {
            max-width: 680px;
            padding: 40px 24px;
          }
          h1 {
            margin: 0 0 12px;
            font-size: 48px;
            line-height: 1;
            color: #60a5fa;
          }
          p {
            color: #94a3b8;
            font-size: 18px;
            line-height: 1.6;
          }
          code {
            color: #67e8f9;
          }
        </style>
      </head>
      <body>
        <main>
          <h1>VoiceIQ</h1>
          <p>Call Sentiment Intelligence API is running on Vercel.</p>
          <p>Health check: <code>/health</code></p>
        </main>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"})


@app.get("/history")
def history():
    call_history = load_history()
    return jsonify({"summary": summarize_history(call_history), "items": call_history[:20]})


@app.post("/analyze-text")
def analyze_text():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        sentiment_model = load_sentiment_model()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 501

    sentiment = sentiment_model(text)[0]
    return jsonify(
        {
            "transcript": text,
            "sentiment": sentiment.get("label", "NEUTRAL"),
            "confidence": sentiment.get("score", 0.0),
        }
    )


@app.post("/analyze-audio")
def analyze_audio():
    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify({"error": "Missing file field"}), 400

    try:
        text = transcribe_audio(uploaded_file)
        sentiment_model = load_sentiment_model()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 501

    sentiment = sentiment_model(text)[0]

    return jsonify(
        {
            "filename": uploaded_file.filename,
            "transcript": text,
            "sentiment": sentiment.get("label", "NEUTRAL"),
            "confidence": sentiment.get("score", 0.0),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
