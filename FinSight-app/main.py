from curl_cffi import request
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
import yfinance as yf 

def get_historical_data(stock_code: str) -> pd.DataFrame:
    try:
        # Example: Fetching last 60 days of data
        # Note: NSE stocks on Yahoo Finance usually end in '.NA' (e.g., EGAD.NA)
        ticker = yf.Ticker(f"{stock_code}.NA")
        hist = ticker.history(period="60d")
        
        if hist.empty:
            return pd.DataFrame()
            
        # Format to match your notebook's expected columns
        df = hist.reset_index()
        df = df.rename(columns={
            'Date': 'Date',
            'Close': 'Day Price',
            'Volume': 'Volume'
        })
        df['Code'] = stock_code
        df['12m High'] = df['Day Price'].max() # Mocking 12m high/low for the API call
        df['12m Low'] = df['Day Price'].min()
        
        return df[['Date', 'Code', 'Day Price', 'Volume', '12m High', '12m Low']]
    except Exception as e:
        print(f"Data fetch error: {e}")
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
        latest_row = latest_row[features]
        # Check if critical features are NaN due to insufficient history
        if latest_row.isnull().values.any():
            raise HTTPException(
            status_code=400, 
            detail=f"Insufficient historical data to calculate technical indicators for {request.stock_code}. Need at least 30-60 days of data."
    )
        
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
