"""
=============================================================================
Fake News Detection - Model Training Script
=============================================================================
This script:
1. Loads the labelled news dataset ('dataset/news.csv').
2. Combines the 'title' and 'text' columns for full context analysis.
3. Splits data into 80% Training set and 20% Testing set.
4. Converts raw text into mathematical vectors using TF-IDF Vectorizer.
5. Trains a Logistic Regression classification model.
6. Evaluates performance using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
7. Saves the trained model and vectorizer into the 'model/' directory.
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------------------------
# Step 1: File Paths Setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "news.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def train_fake_news_detector():
    print("=" * 65)
    print("        FAKE NEWS DETECTION - MODEL TRAINING")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Step 2: Load the Dataset
    # -----------------------------------------------------------------------
    print(f"\n[1/6] Loading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}!")

    df = pd.read_csv(DATASET_PATH)
    print(f"      Total records loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"      Columns found: {list(df.columns)}")

    # -----------------------------------------------------------------------
    # Step 3: Data Preprocessing & Feature Preparation
    # -----------------------------------------------------------------------
    print("\n[2/6] Preprocessing and combining text features...")

    # Fill any null values with an empty string just in case
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')

    # Combine title and text so the model learns from both headlines & paragraphs
    df['combined_content'] = df['title'] + " " + df['text']

    X = df['combined_content']
    y = df['label']

    print(f"      Label breakdown: {dict(y.value_counts())}")

    # -----------------------------------------------------------------------
    # Step 4: Split into Train (80%) and Test (20%) Sets
    # -----------------------------------------------------------------------
    print("\n[3/6] Splitting dataset into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y  # Maintains exact 50/50 Real/Fake ratio in both train & test sets
    )
    print(f"      Training samples : {len(X_train)}")
    print(f"      Testing samples  : {len(X_test)}")

    # -----------------------------------------------------------------------
    # Step 5: Convert Text to Numerical Features using TF-IDF
    # -----------------------------------------------------------------------
    # TF-IDF (Term Frequency - Inverse Document Frequency) measures how important
    # a word is to a document relative to the whole collection of documents.
    # stop_words='english' automatically filters out neutral words like 'the', 'is'.
    # max_df=0.7 ignores words that appear in over 70% of documents (too common).
    print("\n[4/6] Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        ngram_range=(1, 2),  # Analyzes single words ("breaking") and pairs ("breaking news")
        max_features=50000   # Keeps top 50,000 most informative features
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"      Vocabulary size: {X_train_tfidf.shape[1]} features extracted")

    # -----------------------------------------------------------------------
    # Step 6: Train Logistic Regression Classifier
    # -----------------------------------------------------------------------
    # Logistic Regression is one of the most effective and reliable algorithms
    # for high-dimensional text classification. It computes clear probabilities.
    print("\n[5/6] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_train_tfidf, y_train)
    print("      Model training complete!")

    # -----------------------------------------------------------------------
    # Step 7: Model Evaluation (Testing Set)
    # -----------------------------------------------------------------------
    print("\n[6/6] Evaluating model performance on unseen test data...")
    y_pred = model.predict(X_test_tfidf)

    # Compute key metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='REAL')
    rec = recall_score(y_test, y_pred, pos_label='REAL')
    f1 = f1_score(y_test, y_pred, pos_label='REAL')

    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

    print("-" * 65)
    print(f"  ACCURACY  : {acc * 100:.2f}%")
    print(f"  PRECISION : {prec * 100:.2f}% (for REAL news)")
    print(f"  RECALL    : {rec * 100:.2f}% (for REAL news)")
    print(f"  F1-SCORE  : {f1 * 100:.2f}%")
    print("-" * 65)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(f"                 Predicted FAKE   Predicted REAL")
    print(f"  Actual FAKE :       {cm[0][0]:<14} {cm[0][1]}")
    print(f"  Actual REAL :       {cm[1][0]:<14} {cm[1][1]}")
    print("-" * 65)

    # -----------------------------------------------------------------------
    # Step 8: Save Model and Vectorizer to Disk
    # -----------------------------------------------------------------------
    print("\nSaving trained model and vectorizer...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)
    print(f"  [SAVED] Model file      : {MODEL_PATH}")
    print(f"  [SAVED] Vectorizer file : {VECTORIZER_PATH}")

    # -----------------------------------------------------------------------
    # Step 9: Quick Sanity Verification
    # -----------------------------------------------------------------------
    print("\nRunning quick test with sample input...")
    sample_text = ["NASA Mars Rover discovers evidence of ancient lakebed with organic molecules."]
    sample_vec = tfidf.transform(sample_text)
    sample_pred = model.predict(sample_vec)[0]
    sample_probs = model.predict_proba(sample_vec)[0]
    prob_real = sample_probs[list(model.classes_).index('REAL')] * 100
    prob_fake = sample_probs[list(model.classes_).index('FAKE')] * 100

    print(f"  Sample Input: \"{sample_text[0]}\"")
    print(f"  Predicted Verdict : {sample_pred}")
    print(f"  Confidence: Real = {prob_real:.1f}%, Fake = {prob_fake:.1f}%")
    print("\nTraining workflow completed successfully! Ready for Flask backend.")


if __name__ == "__main__":
    train_fake_news_detector()
