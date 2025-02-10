import pandas as pd
import numpy as np
import nltk
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Sample dataset (Replace with actual dataset)
data = {
    "text": [
        "The stock market is performing well today!",
        "I am happy with my investments.",
        "The economic downturn is affecting the stock market...",
        "Bad news for investors as stocks plummet!!!",
        "Great earnings report boosts investor confidence.",
        "The company is facing financial troubles.",
        "Bullish trend observed in tech stocks.",
        "Market sentiment is negative due to inflation concerns."
    ],
    "sentiment": ["positive", "positive", "negative", "negative", "positive", "negative", "positive", "negative"]
}

df = pd.DataFrame(data)

# **Clean Text Function**
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = word_tokenize(text)  # Tokenize text
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["sentiment"], test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(pipeline, "sentiment_model.pkl")
print("Model saved as 'sentiment_model.pkl'")
