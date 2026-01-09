import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transformers import pipeline

# --- Config ---
WATCH_FOLDER = "daily_logs/"
REPORT_FOLDER = "reports/"
CHART_FOLDER = "charts/"
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

# --- Load Models ---
@st.cache(allow_output_mutation=True)
def load_models():
    sentiment_analyzer = pipeline("sentiment-analysis")
    emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
    return sentiment_analyzer, emotion_analyzer

sentiment_analyzer, emotion_analyzer = load_models()

# --- Auto-monitoring ---
class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".txt"):
            return
        with open(event.src_path, "r") as f:
            text = f.read()
        sentences = text.split(". ")

        # Analysis
        sentiment_results = [sentiment_analyzer(s)[0] for s in sentences]
        emotion_results = [emotion_analyzer(s)[0] for s in sentences]

        # Productivity score
        positive = sum(1 for r in sentiment_results if r['label']=='POSITIVE')
        negative = sum(1 for r in sentiment_results if r['label']=='NEGATIVE')
        productivity_score = max(0, min(100, 50 + (positive - negative)*10))

        # Save CSV report
        report = pd.DataFrame({
            "Sentence": sentences,
            "Sentiment": [r['label'] for r in sentiment_results],
            "Sentiment_Score": [r['score'] for r in sentiment_results],
            "Emotion": [r['label'] for r in emotion_results],
            "Emotion_Score": [r['score'] for r in emotion_results]
        })
        report_file = os.path.join(REPORT_FOLDER, os.path.basename(event.src_path).replace(".txt","_report.csv"))
        report.to_csv(report_file, index=False)

        # Emotion chart
        plt.figure(figsize=(6,4))
        sns.countplot(x=[r['label'] for r in emotion_results])
        plt.title(f"Emotion Distribution - {os.path.basename(event.src_path)}")
        chart_file = os.path.join(CHART_FOLDER, os.path.basename(event.src_path).replace(".txt","_chart.png"))
        plt.savefig(chart_file)
        plt.close()

        print(f"Processed {event.src_path} | Productivity Score: {productivity_score}/100")

# Start observer in background
observer = Observer()
observer.schedule(FileHandler(), path=WATCH_FOLDER, recursive=False)
observer.start()

# --- Streamlit Dashboard ---
st.title("Unified AI Productivity Dashboard")

# Select which report to display
reports = [f for f in os.listdir(REPORT_FOLDER) if f.endswith("_report.csv")]
selected_report = st.selectbox("Select a report to view:", reports)

if selected_report:
    df = pd.read_csv(os.path.join(REPORT_FOLDER, selected_report))
    st.subheader("Detailed Analysis")
    st.dataframe(df)

    st.subheader("Productivity Score")
    positive = sum(df['Sentiment'] == 'POSITIVE')
    negative = sum(df['Sentiment'] == 'NEGATIVE')
    productivity_score = max(0, min(100, 50 + (positive - negative)*10))
    st.metric("Productivity Score", productivity_score)

    st.subheader("Emotion Distribution")
    chart_path = os.path.join(CHART_FOLDER, selected_report.replace("_report.csv","_chart.png"))
    st.image(chart_path)

# Keep Streamlit running while observer runs
st.write("Monitoring `daily_logs/` for new files... Auto-updates when new files arrive.")
