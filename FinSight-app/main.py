from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="FinSight NSE Predictor API")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and expected feature list
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

# 1. Define the exact input the Frontend will send
class StockRequest(BaseModel):
    stock_code: str

# 2. Mock function to fetch historical data (Replace with your DB/API query)
def get_historical_data(stock_code: str) -> pd.DataFrame:
    """
    TODO: Replace this with your actual database or API call.
    You need to fetch the last ~30-60 days of raw data for the stock_code.
    Columns needed: 'Date', 'Code', 'Day Price', 'Volume', '12m High', '12m Low'
    """
    # Example: df = pd.read_sql(f"SELECT * FROM stocks WHERE Code='{stock_code}' ORDER BY Date DESC LIMIT 60")
    # For now, returning an empty dataframe to show structure
    return pd.DataFrame() 

# 3. Feature Engineering Pipeline (Extracted from your Notebook)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(by='Date')

        
    # Rolling averages and volatility
    for window in [5, 10, 20]:
        df[f'MA_{window}'] = df['Day Price'].rolling(window, min_periods=1).mean()
        df[f'Vol_{window}'] = df['Day Price'].rolling(window, min_periods=2).std()
        
    # Price ratios & Returns
    df['Price_to_High'] = df['Day Price'] / df['12m High']
    df['Price_to_Low'] = df['Day Price'] / df['12m Low']
    df['Daily_Return'] = df['Day Price'].pct_change()
    
    # EMAs
    for window in [5, 10, 20]:
        df[f'EMA_{window}'] = df['Day Price'].ewm(span=window, adjust=False).mean()
        
    # RSI (14-day)
    delta = df['Day Price'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Day Price'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Day Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Momentum
    df['Momentum_10'] = df['Day Price'].pct_change(10)
    
    # Volume features
    df['Volume_MA_5'] = df['Volume'].rolling(5, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    
    return df

@app.post("/predict")
def predict_stock(request: StockRequest):
    try:
        # Step 1: Fetch historical data for the stock
        hist_df = get_historical_data(request.stock_code)
        if hist_df.empty:
            raise HTTPException(status_code=404, detail="Historical data not found for this stock.")
            
        # Step 2: Engineer features
        df_features = engineer_features(hist_df)
        
        # Step 3: Isolate the latest row (today's data) for prediction
        latest_row = df_features.iloc[[-1]] 
        
        # Step 4: Ensure columns match model expectations and handle any NaNs from lags
        latest_row = latest_row[features].fillna(0) # Fill NaNs if early lags are missing
        
        # Step 5: Predict
        prediction = model.predict(latest_row)[0]
        probabilities = model.predict_proba(latest_row)[0]
        prob_up = probabilities[1] 
        
        return {
            "status": "success",
            "stock_code": request.stock_code,
            "direction": "UP 📈" if prediction == 1 else "DOWN 📉",
            "confidence": f"{round(prob_up * 100, 2)}%",
            "message": "Model is confident in this prediction." if prob_up > 0.65 else "Model is uncertain. Proceed with caution."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "FinSight API is running! Visit /docs to see the interactive API documentation."}
