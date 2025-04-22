from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return "✅ Heart Disease Prediction API is running."

model = pickle.load(open("model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("Received data:", data)

    # Required keys
    input_features = [float(data[k]) for k in ['age', 'sex', 'cp', 'chol', 'trestbps', 'thalach']]

    prediction = model.predict([input_features])[0]
    return jsonify({
        "prediction": int(prediction),
        "risk": "High" if prediction == 1 else "Low"
    })

if __name__ == '__main__':
    app.run(debug=True)
