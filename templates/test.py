import joblib

# Load trained model and vectorizer
vectorizer, model = joblib.load("sentiment_model.pkl")

def predict_sentiment(text):
    """Predict the sentiment of a given text input."""
    cleaned_text = [" ".join(text.lower().split())]  # Simple text cleaning
    vectorized_text = vectorizer.transform(cleaned_text)  # Convert to numerical format
    prediction = model.predict(vectorized_text)[0]  # Predict sentiment
    return prediction

# **🔹 Testing with Sample Inputs**
test_sentences = [
    "Stock prices are rising significantly today!",
    "The company is facing major financial losses.",
    "I am optimistic about the market trends.",
    "The market crash is causing panic among investors.",
    "This is the best time to invest in stocks!"
]

# **🔹 Running Predictions**
print("\n🔍 Testing Sentiment Analysis Model...\n")
for sentence in test_sentences:
    result = predict_sentiment(sentence)
    print(f"Text: {sentence} ➝ Predicted Sentiment: {result}")

print("\n✅ Model testing complete!")
