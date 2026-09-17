import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd

# Use absolute paths so Render can find files regardless of where it runs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(app)


# 1. Load Model & Features Using Absolute Paths

print("Loading FinSight Model and Features...")
try:
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    features_path = os.path.join(BASE_DIR, 'features.pkl')
    
    model = joblib.load(model_path)
    expected_features = joblib.load(features_path)
    print("✅ Model loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: Model files not found.")
    exit(1)

# 2. Define the API endpoints

@app.route('/')
def home():
    # Serve the frontend using the absolute base directory
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        df = pd.DataFrame([data])

        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            return jsonify({"error": "Missing features", "missing": list(missing_features)}), 400

        for col in expected_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if df[expected_features].isnull().any().any():
            return jsonify({"error": "Invalid data types"}), 400

        df = df[expected_features]
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "label": "Price Up 📈" if prediction == 1 else "Price Down/Flat 📉",
            "confidence": {
                "Price_Up_Percent": round(float(probabilities[1]) * 100, 2),
                "Price_Down_Percent": round(float(probabilities[0]) * 100, 2)
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 3. Run the Server 

if __name__ == '__main__':
    # Use Render Provided port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) # debug=False for production!
