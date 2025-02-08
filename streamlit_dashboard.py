import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import seaborn as sns

st.title("Real-Time Dashboard with Matplotlib")

# Live updating plot
placeholder = st.empty()
df = pd.read_csv('emails_with_sentiment.csv')

while True:
    fig, ax = plt.subplots()
    tone_counts = df['tone'].value_counts()
    tone_counts.plot(kind="bar", colormap="coolwarm", edgecolor="black", ax=ax)

    # Labels and title
    ax.set_xlabel("Tone Category")
    ax.set_ylabel("Count of Messages")
    ax.set_title("Distribution of Tones in Messages")
    ax.set_xticklabels(tone_counts.index, rotation=45)

    placeholder.pyplot(fig)
    time.sleep(2)  # Update every 2 seconds