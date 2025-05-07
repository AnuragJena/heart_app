from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from attention_module import AdaptiveFeatureAttention  # Required for loading

# Load components
model = joblib.load("attention_model.pkl")
attention = joblib.load("attention_layer.pkl")
scaler = joblib.load("scaler.pkl")

# Expected input features
expected_features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "bp_variability", "bmi", "diabetes"
]

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Heart Disease Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate required fields
        if not all(feature in data for feature in expected_features):
            return jsonify({"error": "Missing one or more required input features."})

        input_data = [float(data[feature]) for feature in expected_features]
        input_scaled = scaler.transform([input_data])
        input_attn = attention.transform(input_scaled)
        prediction = model.predict(input_attn)[0]
        confidence = model.predict_proba(input_attn)[0][int(prediction)]
        #Added code for Probability 07-05-2025

        return jsonify({
            "prediction": int(prediction),
            "risk": "High" if prediction == 1 else "Low",
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
