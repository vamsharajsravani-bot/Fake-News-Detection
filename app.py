from flask import Flask, render_template, request
import pickle
import pandas as pd
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------------------
# Load Model and Vectorizer
# -------------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    print("Error loading model/vectorizer:", e)
    model = None
    vectorizer = None

# -------------------------------
# Load Dataset
# -------------------------------
try:
    df = pd.read_csv("news.csv")

    if "text" not in df.columns:
        raise ValueError("news.csv must contain a 'text' column.")

except Exception as e:
    print("Error loading dataset:", e)
    df = pd.DataFrame(columns=["text"])

# -------------------------------
# Clean Text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# Vectorize Known News
# -------------------------------
if model is not None and vectorizer is not None and not df.empty:
    df["cleaned_text"] = df["text"].apply(clean_text)
    known_vectors = vectorizer.transform(df["cleaned_text"])
else:
    known_vectors = None

# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Prediction
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if model is None or vectorizer is None:
        return render_template(
            "index.html",
            prediction="Model not loaded",
            confidence="N/A"
        )

    news_text = request.form.get("news", "").strip()

    if news_text == "":
        return render_template(
            "index.html",
            prediction="Please enter news text.",
            confidence="N/A"
        )

    cleaned_input = clean_text(news_text)

    vector_input = vectorizer.transform([cleaned_input])

    prediction = "Data not found"
    confidence = "N/A"

    if known_vectors is not None:

        similarities = cosine_similarity(vector_input, known_vectors)
        max_similarity = similarities.max()

        threshold = 0.70

        if max_similarity >= threshold:

            pred = model.predict(vector_input)[0]

            # Convert numeric labels to text
            if pred == 1:
                prediction = "REAL"
            elif pred == 0:
                prediction = "FAKE"
            else:
                prediction = str(pred)

            confidence = f"{max_similarity * 100:.2f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

# -------------------------------
# Run Application
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)

