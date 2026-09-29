# 📰 Fake News Detector

A Machine Learning project that classifies news text as fake or real using:

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## Project Flow

Dataset → Text Cleaning → TF-IDF → Logistic Regression → Prediction → Streamlit UI

## Dataset Format

Create `fake_news_dataset.csv` with at least these columns:

```csv
text,label
"Your news text here",real
"Another news example",fake
```

If your dataset has `title` instead of `text`, the training script accepts `title`.

The label can be values such as:

- `fake` / `real`
- `false` / `true`
- `0` / `1`

## Installation

Open Command Prompt or VS Code terminal:

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This creates:

- `model.pkl`
- `tfidf_vectorizer.pkl`
- `test_results/test_predictions.csv`

The terminal also displays accuracy, classification report and confusion matrix.

## Run Streamlit

```bash
streamlit run app.py
```

The browser will open the application.

## Features

### Single prediction
Enter a headline/news text and get:

- Fake/Real prediction
- Confidence percentage

### Batch prediction
Upload a CSV with a `text` or `title` column and download predictions as CSV.

## Important

The confidence value is the model's estimated class probability, not proof that the news is factually true. Real-world fact checking should use reliable sources and human verification.
