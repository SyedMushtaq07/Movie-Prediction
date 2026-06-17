from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch

app = Flask(__name__)
CORS(app)

print("Loading BERT model...")

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

print("Model Loaded Successfully!")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data.get("text", "")

    if text.strip() == "":
        return jsonify({
            "error": "Please enter a review"
        }), 400

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    negative_score = float(probs[0]) * 100
    positive_score = float(probs[1]) * 100

    if positive_score >= negative_score:
        sentiment = "POSITIVE"
        confidence = positive_score
    else:
        sentiment = "NEGATIVE"
        confidence = negative_score

    # Detect mixed opinions
    contrast_words = [
        "but",
        "however",
        "though",
        "although",
        "yet",
        "while"
    ]

    text_lower = text.lower()

    if any(word in text_lower for word in contrast_words):

        if sentiment == "POSITIVE":
            positive_score = 60.0
            negative_score = 40.0
            confidence = 60.0

        else:
            positive_score = 40.0
            negative_score = 60.0
            confidence = 60.0

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "positive_score": round(positive_score, 2),
        "negative_score": round(negative_score, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)