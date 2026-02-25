Call Sentiment AI
Overview
Call Sentiment AI is a simple web application that analyzes customer call recordings by converting speech into text and detecting the sentiment of the conversation. 
The system classifies calls as positive, negative, or neutral to help understand customer feedback.

Features:-
Upload audio files for analysis
Transcribe audio into text
Perform sentiment analysis on transcripts
Display sentiment result clearly in the interface
Simple and responsive UI

How It Works:-
The user uploads an audio file.
The system converts the audio into text using a speech recognition model.
The text is analyzed using a sentiment analysis model.
The result is shown as positive, negative, or neutral.

Technologies Used:-
-Python
-Streamlit
-Whisper (speech-to-text)
-PyTorch
-Transformers
-FFmpeg
-Git and GitHub

Project Structure:-
CallSentimentAI/
│
├── app.py
├── requirements.txt
├── README.md
└── other project files

Installation:-
Clone the repository:
git clone https://github.com/Siyaphor/CallSentimentAI.git
cd CallSentimentAI

Install dependencies:
pip install -r requirements.txt
Ensure FFmpeg is installed and available in PATH.

Run the Application
streamlit run app.py
Open the local URL shown in the terminal.

Use Case:-
This project can help analyze customer support calls to understand customer sentiment and improve service quality.

Future Upgrades:-
-Real-time call monitoring
-Live microphone input
-Dashboard with analytics
-Call history tracking
-Emotion detection beyond sentiment
-Integration with support systems

Author
-Siya Phor
