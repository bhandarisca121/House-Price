from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model from the model folder
model = joblib.load(os.path.join('model', 'linear_model.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        bedrooms = float(data['Bedrooms'])
        bathrooms = float(data['Bathrooms'])
        sqft = float(data['SquareFootage'])
    except (KeyError, ValueError) as e:
        return jsonify({'error': 'Invalid input'}), 400

    # Assuming your model expects 4 features but now we're only passing 3:
    # You need to either:
    # - retrain your model with only 3 features
    # - OR fill in a default value for the 4th (e.g., LocationScore = 5)
    location_score = 5.0  # Default or mean value
    features = np.array([[bedrooms, bathrooms, sqft, location_score]])

    prediction = model.predict(features)[0]
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
