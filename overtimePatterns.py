import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob 


df = pd.read_csv('emails_with_sentiment.csv')
# Flag late-night emails
df["late_night"] = df["Timestamp"].apply(lambda x: int(pd.to_datetime(x).hour > 20))

# Flag overtime mentions
overtime_keywords = ["working late", "overtime", "extra hours"]
df["overtime_flag"] = df["Body"].apply(lambda x: any(word in x for word in overtime_keywords))

# Plot 1: Distribution of late-night emails
plt.figure(figsize=(8, 6))
df["late_night"].value_counts().plot(kind="bar", color=["skyblue", "orange"])
plt.title("Distribution of Late-Night Emails")
plt.xlabel("Late Night (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
plt.show()

# Plot 2: Distribution of overtime mentions
plt.figure(figsize=(8, 6))
df["overtime_flag"].value_counts().plot(kind="bar", color=["purple", "green"])
plt.title("Distribution of Overtime Mentions")
plt.xlabel("Overtime Mentioned (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
plt.show()

# Plot 3: Late-night emails vs. Overtime mentions
plt.figure(figsize=(10, 6))
crosstab = pd.crosstab(df["late_night"], df["overtime_flag"])
crosstab.plot(kind="bar", stacked=True, color=["lightcoral", "teal"], figsize=(8, 6))
plt.title("Late-Night Emails vs. Overtime Mentions")
plt.xlabel("Late Night (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.legend(["No Overtime Mention", "Overtime Mention"], title="Overtime Flag")
plt.show()