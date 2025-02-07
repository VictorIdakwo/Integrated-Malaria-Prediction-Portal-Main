from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path to your model
model_path = "./Models/Clinical/model/Random_Forest.pkl"
model = joblib.load(model_path)

# Feature list as per your updated data
features = [
    'chill_cold', 'headache', 'fever', 'generalized body pain',
    'abdominal pain', 'Loss of appetite', 'joint pain', 'vomiting',
    'nausea', 'diarrhea'
]

# Root route to render the index.html page
@app.route('/')
def home():
    return render_template('clinical_index.html')  # Make sure 'index.html' is in the templates folder

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Receive the data as JSON
    input_data = {feature: data.get(feature, 0) for feature in features}  # Prepare data for prediction
    input_df = pd.DataFrame([input_data])  # Convert the data into a DataFrame
    prediction = model.predict(input_df)[0]  # Predict the malaria result
    prediction_proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    # Return prediction results as a JSON response
    return jsonify({
        "prediction": int(prediction),
        "prediction_proba": prediction_proba
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
