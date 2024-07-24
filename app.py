from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)
print("Current working directory:", os.getcwd())

# Load the model and scaler with relative paths
model_path = "fish_weight_model.pkl"
scaler_path = "scaler.pkl"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['length1'], data['length2'], data['length3'], data['height'], data['width'], data['species']]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'weight': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

