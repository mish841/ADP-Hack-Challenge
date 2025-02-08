import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('emails_with_sentiment.csv')
# Plot sentiment distribution
sns.countplot(x="sentiment", data=df, palette="viridis")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

sentiment_counts = df.groupby(["Communication Channel", "sentiment"]).size().unstack()

# Plot settings
fig, ax = plt.subplots(figsize=(8, 5))
sentiment_counts.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")

# Labels and title
ax.set_xlabel("Communication Channel")
ax.set_ylabel("Count of Messages")
ax.set_title("Sentiment Distribution by Communication Channel")
ax.legend(title="Sentiment")

# Show the plot
plt.xticks(rotation=0)
plt.show()

tone_counts = df['tone'].value_counts
fig, ax = plt.subplots(figsize=(8, 5))
tone_counts.plot(kind="bar", colormap="coolwarm", edgecolor="black", ax=ax)

# Labels and title
ax.set_xlabel("Tone Category")
ax.set_ylabel("Count of Messages")
ax.set_title("Distribution of Tones in Messages")
ax.set_xticklabels(tone_counts.index, rotation=45)

# Show the plot
plt.show()