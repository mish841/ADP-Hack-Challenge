import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Employee Morale Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
/* Main container */
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 2rem;
    background-color: #F9F9F9;  /* Soft background color */
    margin-bottom: 2rem;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Title and subtitle font styling */
h1, h2 {
    font-family: 'Poppins', sans-serif;
    font-weight: bold;
    color: #333;
}

h2 {
    font-size: 1.8rem;
    color: #555;
}

/* Metric styling */
[data-testid="stMetric"] {
    background-color: #90EE90;  /* Light Green */
    text-align: center;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
    font-family: 'Poppins', sans-serif;
}

[data-testid="stMetric"]:hover {
    transform: scale(1.05);  /* Hover effect */
}

[data-testid="stMetricLabel"] {
    font-size: 16px;
    font-weight: 600;
    color: #333;
}

/* Metric value font styling */
[data-testid="stMetricValue"] {
    font-size: 24px;
    font-weight: 700;
    color: #2d2d2d;
}

/* Tone and emotion colors */
[data-testid="stMetricDeltaIcon-Up"] {
    color: #32CD32;  /* Green for positive */
}

[data-testid="stMetricDeltaIcon-Down"] {
    color: #FF6347;  /* Red for negative */
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #E9F5E1;  /* Soft green for sidebar */
    border-radius: 8px;
    padding: 20px;
}

/* Sidebar font style */
.sidebar h1 {
    font-family: 'Poppins', sans-serif;
    font-size: 1.3rem;
    color: #333;
}

/* Pie and bar chart colors */
.plotly-graph-div .main-svg {
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

/* Additional hover effects */
.stButton button {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    transition: background-color 0.3s ease, transform 0.2s ease;
}

.stButton button:hover {
    background-color: #77DD77;
    transform: translateY(-3px);
}

</style>
""", unsafe_allow_html=True)

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

# Define tone keywords
tone_keywords = {
    "passive_aggressive": ["thanks for", "I guess", "sure, why not", "as usual"],
    "overwhelmed": ["deadline", "urgent", "no time", "stressed", "ASAP", "juggling"],
    "optimistic": ["great", "excited", "right", "we can do this", "looking forward"],
    "relaxed": ["no worries", "take your time", "chill", "whenever"]
}

# Function to detect tone based on keywords
def detect_tone(text):
    text_lower = text.lower()
    for tone, words in tone_keywords.items():
        if any(word in text_lower for word in words):
            return tone
    return "neutral"

# Apply tone detection to the dataset
df["tone"] = df["Body"].apply(detect_tone)

# Save the results
df.to_csv("emails_with_tone.csv", index=False)

# Additional Overtime analysis: flag messages sent after 8 PM as overtime
df["late_night"] = df["Timestamp"].apply(lambda x: int(pd.to_datetime(x).hour > 20))

# Map overtime flag (0, 1) to custom labels
df["late_night_label"] = df["late_night"].map({0: "during working hours", 1: "late-night"})

# Overtime Distribution Pie Chart
overtime_counts = df["late_night_label"].value_counts()

# Dashboard Title
st.title("📊 Employee Morale Dashboard")
st.markdown("""
This dashboard analyzes employee emails, chats, and feedback to understand morale and well-being. 
HR Management can use the filters below to explore the data and take initiative for the workplace environment accordingly.
""")

# Sidebar Filters
with st.sidebar:
    st.title("Filters")
    selected_emotion = st.multiselect("Select Emotions", df["emotion"].unique(), default=df["emotion"].unique())
    selected_tone = st.multiselect("Select Tones", df["tone"].unique(), default=df["tone"].unique())

# Filter the dataset
filtered_df = df[(df["emotion"].isin(selected_emotion)) & (df["tone"].isin(selected_tone))]

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Messages", len(filtered_df))
with col2:
    st.metric("Most Common Emotion", filtered_df["emotion"].mode()[0])
with col3:
    st.metric("Most Common Tone", filtered_df["tone"].mode()[0])

# Visualizations
st.markdown("---")
st.header("Emotion and Tone Distribution")

# Emotion Distribution Pie Chart
emotion_counts = filtered_df["emotion"].value_counts()
fig1 = px.pie(emotion_counts, values=emotion_counts.values, names=emotion_counts.index, 
              title="Emotion Distribution", color_discrete_sequence=px.colors.sequential.Plasma)
st.plotly_chart(fig1, use_container_width=True)

# Tone Distribution Bar Chart
tone_counts = filtered_df["tone"].value_counts()
fig2 = px.bar(tone_counts, x=tone_counts.index, y=tone_counts.values, 
              labels={"x": "Tone", "y": "Count"}, title="Tone Distribution", 
              color=tone_counts.index, color_discrete_sequence=px.colors.sequential.Viridis)
st.plotly_chart(fig2, use_container_width=True)

# Overtime Distribution Pie Chart
st.subheader("Late night messages/emails Distribution")
fig3 = px.pie(overtime_counts, values=overtime_counts.values, names=overtime_counts.index, 
              title="Messages sent during vs outside of working hours", color_discrete_sequence=px.colors.sequential.Plasma)
st.plotly_chart(fig3, use_container_width=True)
