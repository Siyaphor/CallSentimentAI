import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    px = None
    go = None
    HAS_PLOTLY = False

import streamlit as st
from mongo_history import add_history_entry as save_mongo_history_entry
from mongo_history import load_history as load_mongo_history

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# --- Constants
APP_TITLE = "VoiceIQ"
APP_SUBTITLE = "Call Sentiment Intelligence"
ANALYSIS_ENABLED = os.environ.get("ENABLE_ANALYSIS", "true").lower() == "true"
TEMP_AUDIO_DIR = Path("temp_audio")
TEMP_AUDIO_PATH = TEMP_AUDIO_DIR / "incoming_audio"

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🎙️")

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: #050810;
    font-family: 'DM Mono', monospace;
}

[data-testid="stAppViewContainer"] * { color: #e2e8f0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080d1a !important;
    border-right: 1px solid rgba(99,179,237,0.12);
}

[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

section[data-testid="stSidebarContent"] {
    padding: 2rem 1.25rem;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

/* ── Radio buttons → pill nav ── */
[data-testid="stRadio"] > div {
    flex-direction: column;
    gap: 0.4rem;
}

[data-testid="stRadio"] label {
    background: transparent;
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

[data-testid="stRadio"] label:hover {
    background: rgba(99,179,237,0.08);
    border-color: rgba(99,179,237,0.4);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(99,179,237,0.04);
    border: 1.5px dashed rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 2rem;
    transition: border-color 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,179,237,0.5);
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #1a56db, #0ea5e9) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
}

/* ── Text areas ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.03) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
}

/* ── Expanders ── */
details {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(99,179,237,0.1) !important;
    border-radius: 12px !important;
    margin-bottom: 0.6rem !important;
    transition: border-color 0.2s;
}

details:hover {
    border-color: rgba(99,179,237,0.25) !important;
}

summary {
    padding: 0.9rem 1.2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.88rem !important;
    color: #94a3b8 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #0ea5e9 !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(99,179,237,0.1) !important;
    margin: 2rem 0 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: rgba(14,165,233,0.08) !important;
    border: 1px solid rgba(14,165,233,0.2) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    import whisper

    try:
        from transformers import pipeline
    except ImportError:
        from transformers.pipelines import pipeline

    whisper_model = whisper.load_model("base")
    sentiment_model = pipeline("sentiment-analysis")
    return whisper_model, sentiment_model


def load_history() -> List[Dict]:
    try:
        return load_mongo_history()
    except Exception as exc:
        st.warning(f"MongoDB history is unavailable: {exc}")
        return []


def add_history_entry(filename: str, transcript: str, sentiment_label: str, confidence: float):
    try:
        return save_mongo_history_entry(filename, transcript, sentiment_label, confidence)
    except Exception as exc:
        st.error(f"Could not save analysis to MongoDB: {exc}")
        return load_history()


def summarize_history(history: List[Dict]) -> Dict:
    total = len(history)
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
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


# ─────────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────────
SENTIMENT_COLORS = {
    "POSITIVE": {"bg": "linear-gradient(135deg,#065f46,#059669)", "glow": "#059669", "dot": "#34d399"},
    "NEGATIVE": {"bg": "linear-gradient(135deg,#7f1d1d,#dc2626)", "glow": "#dc2626", "dot": "#f87171"},
    "NEUTRAL":  {"bg": "linear-gradient(135deg,#1e293b,#475569)", "glow": "#475569", "dot": "#94a3b8"},
}


def sentiment_badge(label: str, confidence: float = None):
    c = SENTIMENT_COLORS.get(label.upper(), SENTIMENT_COLORS["NEUTRAL"])
    conf_str = f" · {confidence*100:.0f}%" if confidence is not None else ""
    return f"""
    <span style="
        background:{c['bg']};
        box-shadow: 0 0 12px {c['glow']}44;
        color:#fff;
        padding:6px 14px;
        border-radius:999px;
        font-family:'Syne',sans-serif;
        font-weight:700;
        font-size:0.8rem;
        letter-spacing:0.08em;
        text-transform:uppercase;
        display:inline-flex;
        align-items:center;
        gap:6px;
    ">
        <span style="width:7px;height:7px;border-radius:50%;background:{c['dot']};display:inline-block;"></span>
        {label}{conf_str}
    </span>
    """


def stat_card(title: str, value, accent: str, icon: str):
    return f"""
    <div style="
        background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border: 1px solid {accent}33;
        border-top: 2px solid {accent};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position:absolute; top:-20px; right:-10px;
            font-size:4rem; opacity:0.06; line-height:1;
        ">{icon}</div>
        <div style="
            font-family:'Syne',sans-serif;
            font-size:0.7rem;
            letter-spacing:0.12em;
            text-transform:uppercase;
            color:{accent};
            margin-bottom:0.6rem;
            opacity:0.85;
        ">{title}</div>
        <div style="
            font-family:'Syne',sans-serif;
            font-size:2.4rem;
            font-weight:800;
            color:#f1f5f9;
            line-height:1;
        ">{value}</div>
    </div>
    """


def render_cards(stats: dict):
    cols = st.columns(4, gap="medium")
    cards = [
        ("Total Analyzed", stats["total"], "#60a5fa", "📞"),
        ("Positive",       stats["positive"], "#34d399", "😊"),
        ("Neutral",        stats["neutral"],  "#94a3b8", "😐"),
        ("Negative",       stats["negative"], "#f87171", "😟"),
    ]
    for col, (title, val, accent, icon) in zip(cols, cards):
        col.markdown(stat_card(title, val, accent, icon), unsafe_allow_html=True)


def render_charts(stats: dict):
    if stats["total"] == 0:
        st.markdown("""
        <div style="
            text-align:center; padding:4rem;
            border:1px dashed rgba(99,179,237,0.15);
            border-radius:16px; color:#475569;
            font-family:'Syne',sans-serif;
        ">
            <div style="font-size:3rem;margin-bottom:1rem;">🎙️</div>
            <div style="font-size:1rem; font-weight:600;">No calls analyzed yet</div>
            <div style="font-size:0.8rem; margin-top:0.5rem; opacity:0.6;">
                Upload an audio file in the Analyze tab to get started
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not HAS_PLOTLY:
        st.info("Install `plotly` to enable charts.")
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        # Donut chart
        labels = ["Positive", "Neutral", "Negative"]
        values = [stats["positive"], stats["neutral"], stats["negative"]]
        colors = ["#34d399", "#94a3b8", "#f87171"]
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="#050810", width=3)),
            textfont=dict(family="Syne", size=13, color="white"),
            hovertemplate="%{label}: %{value} calls (%{percent})<extra></extra>",
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(
                font=dict(family="Syne", size=12, color="#94a3b8"),
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                xanchor="center", x=0.5, y=-0.05,
            ),
            annotations=[dict(
                text=f"<b>{stats['total']}</b><br><span style='font-size:10px'>CALLS</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(family="Syne", size=22, color="#f1f5f9"),
                xanchor="center",
            )],
        )
        st.markdown("##### Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        # Bar chart with recent trend (last 10)
        history = load_history()
        recent = history[:10][::-1]
        if recent:
            bar_colors = [
                "#34d399" if e.get("sentiment","").upper() == "POSITIVE"
                else "#f87171" if e.get("sentiment","").upper() == "NEGATIVE"
                else "#94a3b8"
                for e in recent
            ]
            fig2 = go.Figure(data=[go.Bar(
                x=[e.get("filename","")[:16]+"…" if len(e.get("filename",""))>16 else e.get("filename","") for e in recent],
                y=[e.get("confidence", 0)*100 for e in recent],
                marker_color=bar_colors,
                marker_line_width=0,
                hovertemplate="%{x}<br>Confidence: %{y:.1f}%<extra></extra>",
            )])
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=60, l=10, r=10),
                xaxis=dict(
                    showgrid=False, tickfont=dict(family="DM Mono", size=10, color="#64748b"),
                    tickangle=-30,
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(99,179,237,0.06)",
                    tickfont=dict(family="DM Mono", size=11, color="#64748b"),
                    title=dict(text="Confidence %", font=dict(family="Syne", size=11, color="#64748b")),
                    range=[0, 100],
                ),
                bargap=0.35,
            )
            st.markdown("##### Confidence — Last 10 Calls")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


def render_history_table(history: List[Dict]):
    if not history:
        st.markdown("""
        <div style="
            text-align:center; padding:3rem;
            border:1px dashed rgba(99,179,237,0.15);
            border-radius:16px; color:#475569;
            font-family:'Syne',sans-serif;
        ">
            <div style="font-size:0.9rem;">No calls in history yet.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for entry in history[:20]:
        created = datetime.fromisoformat(entry["created_at"]).strftime("%b %d, %Y · %H:%M")
        label = entry.get("sentiment", "NEUTRAL").upper()
        conf = entry.get("confidence", 0)
        fname = entry.get("filename", "(unknown)")
        badge = sentiment_badge(label, conf)

        with st.expander(f"🎙️  {fname}  ·  {created}"):
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                {badge}
                <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#475569;">
                    {created}
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.text_area(
                "Transcript",
                value=entry.get("transcript", "(No transcript captured)"),
                height=140,
                key=f"transcript_{entry.get('id')}",
            )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="margin-bottom:2rem;">
            <div style="
                font-family:'Syne',sans-serif;
                font-size:1.5rem;
                font-weight:800;
                background:linear-gradient(135deg,#60a5fa,#0ea5e9);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
                letter-spacing:-0.02em;
            ">🎙️ {APP_TITLE}</div>
            <div style="
                font-family:'DM Mono',monospace;
                font-size:0.7rem;
                color:#475569;
                letter-spacing:0.1em;
                text-transform:uppercase;
                margin-top:0.2rem;
            ">{APP_SUBTITLE}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            font-family:'Syne',sans-serif;
            font-size:0.65rem;
            color:#334155;
            letter-spacing:0.12em;
            text-transform:uppercase;
            margin-bottom:0.5rem;
        ">Navigation</div>
        """, unsafe_allow_html=True)

        page = st.radio("", ["Dashboard", "Analyze Call", "Call History"], label_visibility="collapsed")

        history = load_history()
        stats = summarize_history(history)
        pos_pct = stats["pos_pct"]
        color = "#34d399" if pos_pct >= 60 else "#f87171" if pos_pct < 40 else "#94a3b8"

        st.markdown(f"""
        <div style="
            margin-top:auto;
            margin-bottom:1rem;
            padding:1rem;
            background:rgba(255,255,255,0.02);
            border:1px solid rgba(99,179,237,0.08);
            border-radius:12px;
            margin-top:3rem;
        ">
            <div style="
                font-family:'DM Mono',monospace;
                font-size:0.68rem;
                color:#475569;
                letter-spacing:0.08em;
                text-transform:uppercase;
                margin-bottom:0.8rem;
            ">Health Score</div>
            <div style="
                font-family:'Syne',sans-serif;
                font-size:2rem;
                font-weight:800;
                color:{color};
            ">{pos_pct}%</div>
            <div style="
                font-size:0.72rem;
                color:#475569;
                font-family:'DM Mono',monospace;
            ">positive sentiment</div>
            <div style="
                margin-top:0.8rem;
                height:4px;
                background:rgba(255,255,255,0.05);
                border-radius:2px;
                overflow:hidden;
            ">
                <div style="
                    width:{pos_pct}%;
                    height:100%;
                    background:{color};
                    border-radius:2px;
                    transition:width 1s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            font-family:'DM Mono',monospace;
            font-size:0.65rem;
            color:#1e293b;
            text-align:center;
            padding-top:1rem;
        ">Powered by Whisper · HuggingFace</div>
        """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────
def page_dashboard(history):
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="
            font-family:'Syne',sans-serif;
            font-size:0.7rem;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#334155;
            margin-bottom:0.4rem;
        ">Overview</div>
        <h1 style="
            font-family:'Syne',sans-serif;
            font-size:2.2rem;
            font-weight:800;
            color:#f1f5f9;
            margin:0;
            line-height:1.1;
        ">Call Analytics Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

    stats = summarize_history(history)
    render_cards(stats)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    render_charts(stats)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        font-family:'Syne',sans-serif;
        font-size:1rem;
        font-weight:700;
        color:#94a3b8;
        letter-spacing:0.04em;
        margin-bottom:1rem;
    ">Recent Calls</div>
    """, unsafe_allow_html=True)
    render_history_table(history[:5])


def page_analyze(whisper_model, sentiment_model):
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="
            font-family:'Syne',sans-serif;
            font-size:0.7rem;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#334155;
            margin-bottom:0.4rem;
        ">New Analysis</div>
        <h1 style="
            font-family:'Syne',sans-serif;
            font-size:2.2rem;
            font-weight:800;
            color:#f1f5f9;
            margin:0;
            line-height:1.1;
        ">Analyze a Call</h1>
        <p style="
            font-family:'DM Mono',monospace;
            font-size:0.8rem;
            color:#475569;
            margin-top:0.6rem;
        ">Upload an audio file to transcribe and analyze sentiment.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        font-family:'DM Mono',monospace;
        font-size:0.75rem;
        color:#475569;
        margin-bottom:0.5rem;
    ">Supported formats: mp3 · wav · mp4 · m4a</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload audio", type=["mp3", "wav", "mp4", "m4a"], label_visibility="collapsed")

    if uploaded_file is not None:
        # File info card
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div style="
            background:rgba(99,179,237,0.05);
            border:1px solid rgba(99,179,237,0.15);
            border-radius:12px;
            padding:1rem 1.4rem;
            margin:1rem 0;
            display:flex;
            align-items:center;
            gap:1rem;
        ">
            <div style="font-size:1.8rem;">🎵</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.9rem;color:#e2e8f0;">
                    {uploaded_file.name}
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#475569;">
                    {file_size_kb:.1f} KB · {uploaded_file.type or 'audio'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if TEMP_AUDIO_DIR.exists() and not TEMP_AUDIO_DIR.is_dir():
            TEMP_AUDIO_DIR.unlink()
        TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_AUDIO_PATH.write_bytes(uploaded_file.getvalue())

        if AudioSegment is None:
            st.error("The pydub library is required to process audio files. Install it with `pip install pydub`.")
            return

        with st.spinner("Transcribing audio and analyzing sentiment…"):
            audio = AudioSegment.from_file(TEMP_AUDIO_PATH)
            audio.export("converted.wav", format="wav")
            result = whisper_model.transcribe("converted.wav")
            text = result.get("text", "")
            sentiment = sentiment_model(text)[0]
            label = sentiment.get("label", "NEUTRAL")
            score = sentiment.get("score", 0.0)

        # Results
        st.markdown("""
        <div style="
            font-family:'Syne',sans-serif;
            font-size:0.7rem;
            letter-spacing:0.12em;
            text-transform:uppercase;
            color:#334155;
            margin:1.5rem 0 0.5rem;
        ">Analysis Results</div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 1], gap="medium")

        with col_a:
            st.markdown("""
            <div style="
                font-family:'DM Mono',monospace;
                font-size:0.72rem;
                color:#475569;
                margin-bottom:0.4rem;
            ">Transcript</div>
            """, unsafe_allow_html=True)
            st.text_area("", value=text or "(No speech detected)", height=200, label_visibility="collapsed")

        with col_b:
            c = SENTIMENT_COLORS.get(label.upper(), SENTIMENT_COLORS["NEUTRAL"])
            st.markdown(f"""
            <div style="
                background:{c['bg']};
                box-shadow:0 0 30px {c['glow']}33;
                border-radius:16px;
                padding:1.5rem;
                text-align:center;
                height:100%;
                min-height:180px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                gap:0.6rem;
            ">
                <div style="font-size:2.5rem;">
                    {"😊" if label.upper()=="POSITIVE" else "😟" if label.upper()=="NEGATIVE" else "😐"}
                </div>
                <div style="
                    font-family:'Syne',sans-serif;
                    font-size:1.1rem;
                    font-weight:800;
                    color:#fff;
                    letter-spacing:0.06em;
                    text-transform:uppercase;
                ">{label}</div>
                <div style="
                    font-family:'DM Mono',monospace;
                    font-size:0.75rem;
                    color:rgba(255,255,255,0.7);
                ">{score*100:.1f}% confidence</div>
                <div style="
                    width:60%;
                    height:3px;
                    background:rgba(255,255,255,0.15);
                    border-radius:2px;
                    overflow:hidden;
                    margin-top:0.4rem;
                ">
                    <div style="
                        width:{score*100:.0f}%;
                        height:100%;
                        background:rgba(255,255,255,0.7);
                        border-radius:2px;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        history = add_history_entry(uploaded_file.name, text, label, score)
        st.success("✓ Analysis saved to call history.")


def page_history(history):
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="
            font-family:'Syne',sans-serif;
            font-size:0.7rem;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#334155;
            margin-bottom:0.4rem;
        ">Archive</div>
        <h1 style="
            font-family:'Syne',sans-serif;
            font-size:2.2rem;
            font-weight:800;
            color:#f1f5f9;
            margin:0;
            line-height:1.1;
        ">Call History</h1>
    </div>
    """, unsafe_allow_html=True)

    if history:
        total = len(history)
        st.markdown(f"""
        <div style="
            font-family:'DM Mono',monospace;
            font-size:0.75rem;
            color:#475569;
            margin-bottom:1.5rem;
        ">{total} call{"s" if total != 1 else ""} on record · showing latest 20</div>
        """, unsafe_allow_html=True)

    render_history_table(history)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    page = render_sidebar()
    history = load_history()

    if page == "Dashboard":
        page_dashboard(history)
    elif page == "Analyze Call":
        if not ANALYSIS_ENABLED:
            st.markdown("""
            <div style="margin-bottom:2rem;">
                <div style="
                    font-family:'Syne',sans-serif;
                    font-size:0.7rem;
                    letter-spacing:0.15em;
                    text-transform:uppercase;
                    color:#334155;
                    margin-bottom:0.4rem;
                ">Deployment Mode</div>
                <h1 style="
                    font-family:'Syne',sans-serif;
                    font-size:2.2rem;
                    font-weight:800;
                    color:#f1f5f9;
                    margin:0;
                    line-height:1.1;
                ">Analysis is disabled on this free deployment</h1>
                <p style="
                    font-family:'DM Mono',monospace;
                    font-size:0.8rem;
                    color:#64748b;
                    margin-top:0.8rem;
                    max-width:720px;
                    line-height:1.7;
                ">
                    Whisper and Torch need more memory than Render Free provides. Dashboard and MongoDB call history
                    are still live here; run locally or use a larger instance to analyze new audio files.
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
        whisper_model, sentiment_model = load_models()
        page_analyze(whisper_model, sentiment_model)
    else:
        page_history(history)


if __name__ == "__main__":
    main()
