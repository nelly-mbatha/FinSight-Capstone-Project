from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the FastAPI app
app = FastAPI(title="FinSight NSE Predictor API")

# Load the model and feature list on startup
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

# Define the expected input format (adjust these to match your exact selected_features)
class StockData(BaseModel):
    # Example: Using a dictionary to accept any number of features dynamically
    # In production, you can define each feature explicitly: Lag_1: float, MA_5: float, etc.
    data: dict 

@app.post("/predict")
def predict_stock(data: StockData):
    try:
        # Convert the incoming JSON data into a Pandas DataFrame
        df = pd.DataFrame([data.data])
        
        # Ensure the DataFrame has the exact columns the model expects, in the right order
        df = df[features]
        
        # Make the prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Probability of class '1' (Price Up)
        prob_up = probabilities[1] 
        
        return {
            "status": "success",
            "direction": "UP 📈" if prediction == 1 else "DOWN 📉",
            "confidence": f"{round(prob_up * 100, 2)}%",
            "message": "Model is confident in this prediction." if prob_up > 0.65 else "Model is uncertain. Proceed with caution."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "FinSight API is running! Visit /docs to see the interactive API documentation."}
