from flask import Flask, render_template, request

app = Flask(__name__)

# Function to analyze sentiment (Replace with actual sentiment analysis logic)
def analyze_sentiment(text):
    if "good" in text.lower():
        return "Positive"
    elif "bad" in text.lower():
        return "Negative"
    else:
        return "Neutral"

@app.route("/")
def home():
    return render_template("index.html")  # Ensure index.html is available

@app.route("/predict", methods=["POST"])
def predict():
    user_message = request.form.get("message")  # Get user input
    sentiment = analyze_sentiment(user_message)  # Analyze sentiment
    return f"Predicted Sentiment: {sentiment}"  # Return result

if __name__ == "__main__":
    app.run(debug=True)

