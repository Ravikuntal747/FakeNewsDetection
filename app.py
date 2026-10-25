"""
=============================================================================
Fake News Detection - Flask Backend Application (app.py)
=============================================================================
This backend server:
1. Loads the pre-trained Logistic Regression model ('fake_news_model.pkl')
   and TF-IDF vectorizer ('tfidf_vectorizer.pkl') once into memory.
2. Serves the web interface via the home route ('/').
3. Exposes a REST API endpoint ('/predict') to receive news text,
   validate inputs, generate predictions, and return confidence scores in JSON.
=============================================================================
"""

import os
import joblib
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# 1. Initialize Flask App & Configure Paths
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# ---------------------------------------------------------------------------
# 2. Load Trained Machine Learning Model & TF-IDF Vectorizer
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  LOADING FAKE NEWS DETECTION ML ASSETS...")
print("=" * 55)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}\n"
        "Please run 'python model/train_model.py' first to train and save the model."
    )

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        f"Vectorizer file not found at: {VECTORIZER_PATH}\n"
        "Please run 'python model/train_model.py' first to generate the vectorizer."
    )

# Load the saved model and vectorizer into memory
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
classes_list = list(model.classes_)  # Expected: ['FAKE', 'REAL']

print(f"[OK] Model loaded successfully: {type(model).__name__}")
print(f"[OK] Vectorizer loaded successfully: {type(vectorizer).__name__}")
print(f"[OK] Recognizable classes: {classes_list}")
print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# 3. Route: Home Page
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """
    Renders and displays the main website user interface.
    Flask automatically looks for 'index.html' inside the 'templates/' folder.
    """
    return render_template("index.html")


# ---------------------------------------------------------------------------
# 4. Route: Prediction API Endpoint
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives news text via POST request, runs it through the TF-IDF vectorizer
    and Logistic Regression classifier, and returns the verdict and confidence.
    """
    try:
        # Extract input text from either JSON payload or HTML Form submission
        if request.is_json:
            data = request.get_json()
            news_text = data.get("news_text", "")
        else:
            news_text = request.form.get("news_text", "")

        # -------------------------------------------------------------------
        # Input Validation
        # -------------------------------------------------------------------
        if not news_text or not str(news_text).strip():
            return jsonify({
                "success": False,
                "error": "News content cannot be empty. Please paste a news headline or article body."
            }), 400

        clean_text = str(news_text).strip()

        if len(clean_text) < 10:
            return jsonify({
                "success": False,
                "error": "News text is too short to evaluate. Please provide at least 10 characters."
            }), 400

        # -------------------------------------------------------------------
        # Model Prediction using Loaded ML Assets
        # -------------------------------------------------------------------
        # 1. Transform input text into numerical feature vector
        vectorized_text = vectorizer.transform([clean_text])

        # 2. Predict the label ('REAL' or 'FAKE')
        prediction = model.predict(vectorized_text)[0]

        # 3. Calculate genuine prediction probabilities
        probabilities = model.predict_proba(vectorized_text)[0]
        fake_prob = round(float(probabilities[classes_list.index("FAKE")]) * 100, 2)
        real_prob = round(float(probabilities[classes_list.index("REAL")]) * 100, 2)

        # The confidence score is the probability of the predicted class
        confidence = real_prob if prediction == "REAL" else fake_prob

        # -------------------------------------------------------------------
        # Return Structured JSON Response
        # -------------------------------------------------------------------
        return jsonify({
            "success": True,
            "verdict": prediction,
            "confidence": confidence,
            "probabilities": {
                "REAL": real_prob,
                "FAKE": fake_prob
            },
            "input_preview": clean_text[:120] + ("..." if len(clean_text) > 120 else "")
        }), 200

    except Exception as e:
        # Handle unexpected exceptions gracefully without crashing the server
        return jsonify({
            "success": False,
            "error": f"An error occurred while processing your request: {str(e)}"
        }), 500


# ---------------------------------------------------------------------------
# 5. Route: Health Check (Useful for debugging and testing)
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """
    Returns API server status and verifies model is ready in memory.
    """
    return jsonify({
        "status": "online",
        "model_loaded": True,
        "classes": classes_list
    }), 200


# ---------------------------------------------------------------------------
# 6. Server Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # host='127.0.0.1' makes the server accessible locally in your browser
    # port=5000 is the standard Flask port
    # debug=True automatically reloads changes and shows detailed error traces
    print("Starting Flask web server...")
    print("Access your application at: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
