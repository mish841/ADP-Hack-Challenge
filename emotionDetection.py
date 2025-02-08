import pandas as pd
import torch
from transformers import pipeline
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the dataset
df = pd.read_csv('communication_tone_dataset_v3.csv')
bodyData = df['Body']

# Load the pre-trained emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Function to detect emotions using Hugging Face
def detect_emotion(text):
    result = emotion_classifier(text)[0]
    return result["label"]

# Apply emotion detection to the dataset
df["emotion"] = df["Body"].apply(detect_emotion)

# Save the results to a new CSV file
df.to_csv("emails_with_emotion.csv", index=False)

# Display the first few rows with emotion labels
print(df.head())

# Pie chart visualization
def plot_pie_chart(ax):
    emotion_counts = df['emotion'].value_counts()
    colors = ['#FF6F61', '#6BAED6', '#FFD700', '#77DD77', '#D3D3D3', '#AEC6CF', '#FFB347']  # Aesthetic colors
    
    ax.clear()
    ax.pie(
        emotion_counts,
        labels=emotion_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    ax.set_title("Real-Time Emotion Distribution", fontsize=14, weight='bold')

# Real-time updating pie chart
fig, ax = plt.subplots(figsize=(6, 6))
def update_chart(i):
    plot_pie_chart(ax)

# Create an animation
ani = FuncAnimation(fig, update_chart, interval=2000)

# Show chart
plt.show()