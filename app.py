import streamlit as st
import pandas as pd
import pickle
import re
import io

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "Model files are missing. Run train_model.py first to create "
        "model.pkl and tfidf_vectorizer.pkl."
    )
    st.stop()

st.title("📰 Fake News Detector")
st.write("Enter a news headline or short news text to classify it.")

st.subheader("Single News Check")

news_text = st.text_area(
    "Enter news headline/text:",
    height=150,
    placeholder="Example: Scientists announce a new discovery..."
)

if st.button("🔍 Check News", type="primary"):
    if not news_text.strip():
        st.warning("Please enter a news headline or text.")
    else:
        cleaned = clean_text(news_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        # Probability/confidence is available for Logistic Regression.
        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities) * 100

        label = str(prediction).lower()

        if label in ["fake", "false", "0"]:
            st.error(f"Prediction: FAKE NEWS")
        elif label in ["real", "true", "1"]:
            st.success(f"Prediction: REAL NEWS")
        else:
            st.info(f"Prediction: {str(prediction).upper()}")

        st.metric("Confidence", f"{confidence:.2f}%")

st.divider()

st.subheader("📁 Batch CSV Check")
st.write("Upload a CSV containing a `text` column. A `title` column is also accepted.")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        text_column = None
        if "text" in batch_df.columns:
            text_column = "text"
        elif "title" in batch_df.columns:
            text_column = "title"

        if text_column is None:
            st.error("CSV must contain a 'text' or 'title' column.")
        else:
            batch_df = batch_df.copy()
            cleaned = batch_df[text_column].fillna("").apply(clean_text)
            vectors = vectorizer.transform(cleaned)

            batch_df["prediction"] = model.predict(vectors)

            probabilities = model.predict_proba(vectors)
            batch_df["confidence"] = probabilities.max(axis=1) * 100

            st.dataframe(batch_df, use_container_width=True)

            csv_data = batch_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_data,
                file_name="fake_news_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Could not process the CSV: {e}")

st.caption(
    "Note: This ML model classifies text based on patterns learned from its "
    "training dataset. A high confidence score is not proof that a claim is true."
)
