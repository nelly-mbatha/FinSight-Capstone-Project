from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. LOAD MODEL & FEATURES
# ==========================================
print("🔄 Loading FinSight Model and Features...")
try:
    model = joblib.load('model.pkl')
    expected_features = joblib.load('features.pkl')
    print("✅ Model loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: 'model.pkl' or 'features.pkl' not found. Please ensure they are in the same folder.")
    exit(1)

# ==========================================
# 2. DEFINE API ROUTES
# ==========================================
@app.route('/')
def home():
    return """
    <h1>FinSight NSE Prediction API</h1>
    <p>The model is online and ready to predict short-term stock movements.</p>
    <p><b>Endpoint:</b> POST /predict</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get JSON data from the incoming request
        data = request.get_json()
        
        # 2. Convert to a Pandas DataFrame
        df = pd.DataFrame([data])
        
        # 3. Validate that all 10 required features are present
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            return jsonify({
                "error": "Missing required technical indicators",
                "missing": list(missing_features)
            }), 400
            
        # 4. Defensive Type Coercion (Prevents crashes if client sends strings instead of floats)
        for col in expected_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop if coercion resulted in NaNs (invalid data provided)
        if df[expected_features].isnull().any().any():
            return jsonify({"error": "Invalid data types or NaN values in features"}), 400
            
        # 5. Reorder columns to exactly match how the model was trained
        df = df[expected_features]
        
        # 6. Make the prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # 7. Format the response
        result = {
            "status": "success",
            "prediction": int(prediction),
            "label": "Price Up (1)" if prediction == 1 else "Price Down/Flat (0)",
            "confidence": {
                "Price_Up_Percent": round(float(probabilities[1]) * 100, 2),
                "Price_Down_Percent": round(float(probabilities[0]) * 100, 2)
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 3. RUN THE SERVER
# ==========================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
