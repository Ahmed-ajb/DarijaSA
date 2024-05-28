from flask import Flask, request, jsonify, render_template
from joblib import load

app = Flask(__name__)

# Load the pre-trained model
loaded_model = load('DarijaAM.joblib')

# Define the sentiment analysis function
def analyze_sentiment(sentences):
    predictions = loaded_model.predict(sentences)
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentences = data['sentences']
    
    # Analyze the sentiment of the sentences
    predictions = analyze_sentiment(sentences)
    
    # Generate full sentence responses
    results = []
    for sentence, prediction in zip(sentences, predictions):
        sentiment = "positive" if prediction == 0 else "negative"
        results.append(f"The sentence '{sentence}' is {sentiment}.")
    
    response = {
        'results': results
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
