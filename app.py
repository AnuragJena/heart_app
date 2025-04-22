from flask import Flask, request, jsonify
import joblib
import numpy as np

# === Load trained components ===
model = joblib.load("attention_model.pkl")
attention = joblib.load("attention_layer.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "Heart Disease Predictor API (with Attention Model) is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_features = [float(data[k]) for k in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                                                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                                                   'ca', 'thal', 'bp_variability', 'bmi', 'diabetes']]

        # Preprocess input
        X_scaled = scaler.transform([input_features])
        X_attn = attention.transform(X_scaled)

        prediction = model.predict(X_attn)[0]

        return jsonify({
            "prediction": int(prediction),
            "risk": "High" if prediction == 1 else "Low"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
