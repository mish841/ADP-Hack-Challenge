from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from textblob import TextBlob 

df = pd.read_csv('communication_tone_dataset_v3.csv')
bodyData = df['Body']

def get_sentiment():
    
    # Initialize an empty list to store sentiment results
    sentiments = []
    
    # Loop through each body and perform sentiment analysis
    for body in bodyData: 
        analysis = TextBlob(body)
        
        # Determine sentiment based on polarity
        if analysis.sentiment.polarity > 0:
            sentiments.append("positive")
        elif analysis.sentiment.polarity < 0:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")
    
    # Add sentiment labels to the dataframe
    df["sentiment"] = sentiments
    
    # Display the first few rows with sentiment labels
    print(df.head())
    
    # Save the results to a new CSV file
    df.to_csv("emails_with_sentiment.csv", index=False)
    
    return df


tone_keywords = {
    "passive_aggressive": ["thanks for", "I guess", "sure, why not", "as usual"],
    "overwhelmed": ["deadline", "urgent", "no time", "stressed", "ASAP"],
    "optimistic": ["great", "excited", "right", "we can do this", "looking forward"],
    "relaxed": ["no worries", "take your time", "chill", "whenever"]
}

def detect_tone(text):
    text_lower = text.lower()
    for tone, words in tone_keywords.items():
        if any(word in text_lower for word in words):
            return tone
    return "neutral"  # Default if no tone keywords are found

df["tone"] = df["Body"].apply(detect_tone)
print(get_sentiment())
print(df)
# Pie chart visualization
def plot_pie_chart(ax):
    tone_counts = df['tone'].value_counts()
    colors = ['#FF6F61', '#6BAED6', '#FFD700', '#77DD77', '#D3D3D3']  # Aesthetic colors
    
    ax.clear()
    ax.pie(
        tone_counts,
        labels=tone_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}

    )
    ax.set_title("Real-Time Tone Distribution", fontsize=14, weight='bold')

# Real-time updating pie chart
fig, ax = plt.subplots(figsize=(6, 6))
def update_chart(i):
    plot_pie_chart(ax)

# Create an animation
ani = FuncAnimation(fig, update_chart, interval=2000)

# Show chart
plt.show()