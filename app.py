import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    px = None  # type: ignore
    HAS_PLOTLY = False

import streamlit as st
import whisper
from pydub import AudioSegment
from transformers import pipeline

# --- Constants
APP_TITLE = "Sentiment Analyzer"
HISTORY_PATH = Path("call_history.json")
TEMP_AUDIO_DIR = Path("temp_audio")
TEMP_AUDIO_PATH = TEMP_AUDIO_DIR / "incoming_audio"


st.set_page_config(page_title=APP_TITLE, layout="wide")

# --- Utils
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    sentiment_model = pipeline("sentiment-analysis")
    return whisper_model, sentiment_model


def load_history() -> List[Dict]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            return []
    return []


def save_history(history: List[Dict]):
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def add_history_entry(filename: str, transcript: str, sentiment_label: str, confidence: float):
    history = load_history()
    entry = {
        "id": datetime.utcnow().isoformat(),
        "filename": filename,
        "transcript": transcript,
        "sentiment": sentiment_label,
        "confidence": confidence,
        "created_at": datetime.utcnow().isoformat(),
    }
    history.insert(0, entry)
    save_history(history)
    return history


def summarize_history(history: List[Dict]) -> Dict:
    total = len(history)
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for item in history:
        label = item.get("sentiment", "NEUTRAL").upper()
        if label in counts:
            counts[label] += 1
    return {
        "total": total,
        "positive": counts["POSITIVE"],
        "negative": counts["NEGATIVE"],
        "neutral": counts["NEUTRAL"],
    }


def render_cards(stats: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Calls", stats["total"])
    col2.metric("Positive Calls", stats["positive"])
    col3.metric("Neutral Calls", stats["neutral"])
    col4.metric("Negative Calls", stats["negative"])


def render_pie_chart(stats: dict):
    if stats["total"] == 0:
        st.info("No call history yet. Upload a call in the 'Analyze Call' tab to get started.")
        return

    if not HAS_PLOTLY:
        st.warning(
            "Plotly is not installed, so the sentiment distribution chart cannot be shown. "
            "Install `plotly` or run `pip install -r requirement.txt` to enable it."
        )
        st.markdown(
            f"- **Positive:** {stats['positive']}\n"
            f"- **Neutral:** {stats['neutral']}\n"
            f"- **Negative:** {stats['negative']}"
        )
        return

    labels = ["Positive", "Neutral", "Negative"]
    values = [stats["positive"], stats["neutral"], stats["negative"]]
    fig = px.pie(names=labels, values=values, title="Sentiment Distribution", hole=0.4)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def render_history_table(history: list[dict]):
    if not history:
        st.info("No history yet. Process calls in the 'Analyze Call' tab to populate history.")
        return

    for entry in history[:20]:
        created = datetime.fromisoformat(entry["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
        sentiment_label = entry.get("sentiment", "NEUTRAL")
        badge_color = {
            "POSITIVE": "#059669",
            "NEUTRAL": "#6b7280",
            "NEGATIVE": "#dc2626",
        }.get(sentiment_label.upper(), "#6b7280")

        with st.expander(f"{entry.get('filename','(unknown)')} — {created}"):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**Transcript**")
                st.text_area(
                    "",
                    value=entry.get("transcript", "(No transcript captured)"),
                    height=140,
                    key=f"transcript_{entry.get('id')}",
                )
            with cols[1]:
                st.markdown(
                    f"<div style='padding:10px 14px; border-radius:12px; background:{badge_color}; color:white; display:inline-block; font-weight:600;'>{sentiment_label}</div>",
                    unsafe_allow_html=True,
                )


def main():
    st.markdown(
        """
        <style>
        /* App background */
        [data-testid="stAppViewContainer"] {
            background: #f7fafc;
            color: #0f172a;
        }

        /* Force readable text in the app area */
        [data-testid="stAppViewContainer"] * {
            color: #0f172a !important;
        }

        /* Sidebar background + text (keep sidebar white text) */
        [data-testid="stSidebar"] {
            background: #0f172a;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        /* Expander header and buttons */
        button[aria-expanded] {
            color: #0f172a !important;
            font-weight: 600;
        }

        /* Make text areas easier to read */
        .stTextArea textarea {
            background: #ffffff !important;
            color: #111827 !important;
        }

        /* Ensure headings are visible */
        h1, h2, h3, h4, h5, h6 {
            color: #0f172a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"# {APP_TITLE}")
    st.write("Help Desk Call Sentiment Analytics")

    page = st.sidebar.radio("Navigation", ["Dashboard", "Analyze Call", "Call History"])

    whisper_model, sentiment_model = load_models()
    history = load_history()

    if page == "Dashboard":
        stats = summarize_history(history)
        render_cards(stats)

        st.markdown("---")
        left, right = st.columns([2, 3])
        with left:
            render_pie_chart(stats)
        with right:
            st.subheader("Recent Calls")
            render_history_table(history)

    elif page == "Analyze Call":
        st.header("Analyze a new call")
        uploaded_file = st.file_uploader("Upload audio", type=["mp3", "wav", "mp4", "m4a"])
        if uploaded_file is not None:
            # Ensure temp_audio is a directory; on Windows it can clash if a file with the same name exists.
            if TEMP_AUDIO_DIR.exists() and not TEMP_AUDIO_DIR.is_dir():
                TEMP_AUDIO_DIR.unlink()
            TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            TEMP_AUDIO_PATH.write_bytes(uploaded_file.getvalue())

            with st.spinner("Processing audio and running sentiment analysis…"):
                # Convert to wav
                audio = AudioSegment.from_file(TEMP_AUDIO_PATH)
                audio.export("converted.wav", format="wav")

                # Transcribe
                result = whisper_model.transcribe("converted.wav")
                text = result.get("text", "")

                # Sentiment
                sentiment = sentiment_model(text)[0]
                label = sentiment.get("label", "NEUTRAL")
                score = sentiment.get("score", 0.0)

            st.success("Call analysis completed.")

            st.subheader("Transcript")
            st.text_area("", value=text or "(No transcript detected)", height=180)

            st.subheader("Sentiment")
            badge_color = {
                "POSITIVE": "#059669",
                "NEUTRAL": "#6b7280",
                "NEGATIVE": "#dc2626",
            }.get(label.upper(), "#6b7280")
            st.markdown(
                f"<div style='padding:10px 14px; border-radius:12px; background:{badge_color}; color:white; display:inline-block; font-weight:600;'>{label} (Confidence: {score:.2f})</div>",
                unsafe_allow_html=True,
            )

            history = add_history_entry(uploaded_file.name, text, label, score)
            st.success("Call analysis saved to history.")

    else:  # Call History
        st.header("Call History")
        render_history_table(history)


if __name__ == "__main__":
    main()
