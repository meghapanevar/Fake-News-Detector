import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET = "fake_news_dataset.csv"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv(DATASET)

# Expected columns: text and label.
# If your dataset uses title instead of text, rename title -> text.
if "text" not in df.columns and "title" in df.columns:
    df = df.rename(columns={"title": "text"})

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError(
        "CSV must contain 'text' and 'label' columns. "
        "Example: text,label"
    )

df = df[["text", "label"]].dropna()
df["text"] = df["text"].apply(clean_text)

# Convert labels to strings for a simple, robust app.
df["label"] = df["label"].astype(str).str.strip().str.lower()

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save test results
results = pd.DataFrame({
    "text": X_test.values,
    "actual": y_test.values,
    "predicted": predictions
})
results.to_csv("test_results/test_predictions.csv", index=False)

print("\nSaved:")
print("- model.pkl")
print("- tfidf_vectorizer.pkl")
print("- test_results/test_predictions.csv")
