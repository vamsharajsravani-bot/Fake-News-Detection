import os
import re
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Check Dataset
# -----------------------------
if not os.path.exists("news.csv"):
    raise FileNotFoundError("news.csv not found.")

# Load Dataset
df = pd.read_csv("news.csv")

# Check required columns
required_columns = ["text", "label"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Remove missing values
df = df.dropna(subset=["text", "label"])

# Remove duplicate rows
df = df.drop_duplicates()

# Dataset should not be empty
if df.empty:
    raise ValueError("Dataset is empty.")

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Clean text
df["text"] = df["text"].apply(clean_text)

# Features and Labels
X = df["text"]
y = df["label"]

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# -----------------------------
# Save Model & Vectorizer
# -----------------------------
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully!")