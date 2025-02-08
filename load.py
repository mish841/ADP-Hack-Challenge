import pandas as pd
from textblob import TextBlob 

def get_sentiment():
    # Read the dataset
    df = pd.read_csv('communication_tone_dataset_v3.csv')
    
    # Extract the body text
    bodyData = df['Body']
    
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

# Call the function
print(get_sentiment())
