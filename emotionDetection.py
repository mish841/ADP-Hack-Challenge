import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time
from transformers import pipeline

st.title("Real-Time Dashboard with Matplotlib")

# Load the dataset
df = pd.read_csv('communication_tone_dataset_v3.csv')
bodyData = df['Body']

# Load the pre-trained emotion detection model
st.write("Loading the Hugging Face emotion detection model...")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
st.success("Model loaded successfully!")

# Function to detect emotions using Hugging Face
def detect_emotion(text):
    result = emotion_classifier(text)[0]
    return result["label"]

# Apply emotion detection to the dataset
if "emotion" not in df.columns:
    df["emotion"] = df["Body"].apply(detect_emotion)
    df.to_csv("emails_with_emotion.csv", index=False)

# Live updating bar graph
placeholder1 = st.empty()

# Real-time pie chart visualization
placeholder2 = st.empty()

def plot_bar_chart():
    fig, ax = plt.subplots()
    tone_counts = df['tone'].value_counts()
    tone_counts.plot(kind="bar", colormap="coolwarm", edgecolor="black", ax=ax)

    # Labels and title
    ax.set_xlabel("Tone Category")
    ax.set_ylabel("Count of Messages")
    ax.set_title("Distribution of Tones in Messages")
    ax.set_xticklabels(tone_counts.index, rotation=45)

    return fig

def plot_pie_chart():
    fig, ax = plt.subplots(figsize=(6, 6))
    emotion_counts = df['emotion'].value_counts()
    colors = ['#FF6F61', '#6BAED6', '#FFD700', '#77DD77', '#D3D3D3', '#AEC6CF', '#FFB347']  # Aesthetic colors

    ax.pie(
        emotion_counts,
        labels=emotion_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    ax.set_title("Real-Time Emotion Distribution", fontsize=14, weight='bold')

    return fig

# Continuously update the bar graph and pie chart
while True:
    with placeholder1.container():
        st.write("### Bar Graph: Tone Distribution")
        bar_chart_fig = plot_bar_chart()
        st.pyplot(bar_chart_fig)
    
    with placeholder2.container():
        st.write("### Pie Chart: Emotion Distribution")
        pie_chart_fig = plot_pie_chart()
        st.pyplot(pie_chart_fig)

    time.sleep(2)  # Update every 2 seconds
