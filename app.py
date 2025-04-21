from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "Heart Disease Predictor API is Running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Assume input order: ['age', 'sex', 'cp', 'chol', 'trestbps', 'thalach']
    input_features = [float(data[col]) for col in ['age', 'sex', 'cp', 'chol', 'trestbps', 'thalach']]
    prediction = model.predict([input_features])[0]

    return jsonify({
        "prediction": str(prediction),
        "risk": "High" if prediction == 1 else "Low"
    })

if __name__ == '__main__':
    app.run(debug=True)
