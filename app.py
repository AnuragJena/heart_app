from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return "✅ Heart Disease Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Required keys
    keys = ['age', 'sex', 'cp', 'chol', 'trestbps', 'thalach']
    input_features = [float(data[k]) for k in keys]

    prediction = model.predict([input_features])[0]
    return jsonify({
        "prediction": int(prediction),
        "risk": "High" if prediction == 1 else "Low"
    })

if __name__ == '__main__':
    app.run(debug=True)
