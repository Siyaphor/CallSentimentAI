<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=40&duration=3000&pause=500&color=1F2937&center=true&vCenter=true&width=900&height=120&lines=Call+Sentiment+AI+%7C+Customer+Call+Analysis" alt="Typing SVG" />
</p>

# 📞 Call Sentiment AI - Customer Call Sentiment Analysis Platform

Call Sentiment AI is a Python-based web application built with Streamlit that analyzes customer call recordings using AI-powered speech recognition and sentiment analysis. The system converts customer conversations into text and classifies the sentiment as positive, negative, or neutral.

---

## 🔥 What Call Sentiment AI Does

- Allows users to upload customer call recordings for AI analysis.
- Converts speech into text using Whisper speech recognition.
- Performs sentiment analysis on generated transcripts.
- Detects whether customer sentiment is positive, negative, or neutral.
- Displays results in a clean and responsive web interface.
- Helps businesses understand customer satisfaction and feedback trends.

---

## 🎯 Key Features

- **Audio file upload** for customer call recordings.
- **Speech-to-text transcription** powered by Whisper AI.
- **AI sentiment analysis** using Transformer-based NLP models.
- **Sentiment classification** into Positive, Negative, or Neutral.
- **Simple Streamlit dashboard** with responsive UI.
- **Fast and accurate analysis** for customer support calls.
- **Easy-to-use interface** suitable for non-technical users.

---

## 🧩 Tech Stack

- **Python**
- **Streamlit**
- **Whisper**
- **PyTorch**
- **Transformers**
- **FFmpeg**
- **Git & GitHub**

---

## ⚙️ How the Application Works

- Users upload customer call recordings through the web interface.
- The uploaded audio is processed using Whisper speech recognition.
- Speech is converted into readable text transcripts.
- The transcript is analyzed using a sentiment analysis model.
- The final sentiment result is displayed instantly on the screen.

---

## 📁 Main Application Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Project dependencies |
| `README.md` | Project documentation |
| `audio/` | Uploaded audio recordings |
| `models/` | AI and NLP models |

---

## ⚙️ Installation

```bash
git clone https://github.com/Siyaphor/CallSentimentAI.git
cd CallSentimentAI
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Make sure FFmpeg is installed and available in your system PATH.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🧠 How the System Works

- The application accepts customer call recordings in audio format.
- Whisper AI converts speech into text transcripts.
- NLP sentiment analysis models process the transcript data.
- The application determines customer sentiment polarity.
- Results are displayed with clear sentiment labels and analysis.

---

## 🧭 Project Structure

```text
CallSentimentAI/
│
├── app.py
├── requirements.txt
├── README.md
├── audio/
│   └── uploaded_calls
├── models/
│   └── sentiment_model
├── utils/
│   ├── transcription.py
│   └── sentiment.py
└── assets/
    └── screenshots
```

---

## 💡 Use Cases

- Customer support call analysis
- Customer satisfaction monitoring
- Business feedback analysis
- AI-powered customer experience tracking
- Support center performance monitoring

---

## 🔮 Future Improvements

- Real-time call monitoring
- Live microphone recording support
- Advanced analytics dashboard
- Customer call history management
- Multi-language support
- Emotion detection beyond sentiment analysis
- CRM and support platform integration

---

## ✅ Summary

Call Sentiment AI is an AI-powered customer call analysis platform that combines speech recognition and natural language processing to analyze customer conversations and identify sentiment patterns. The application provides businesses with actionable insights into customer satisfaction through a simple and interactive interface.

---

## 👩‍💻 Author

**Siya Phor**

---

**Built with ❤️ using Python, Streamlit & AI**
