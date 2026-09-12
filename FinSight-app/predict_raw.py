import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load model and features
model = joblib.load('model.pkl')
expected_features = joblib.load('features.pkl')

def clean_raw_data(raw_df):
    """Replicates the exact data cleaning steps from your notebook."""
    df = raw_df.copy()
    
    # 1. Handle string anomalies in numeric columns (e.g., '-' or '#VALUE!')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    if 'Change%' in df.columns:
        df['Change%'] = df['Change%'].astype(str).str.replace('%', '', regex=False).replace(['-', '#VALUE!'], '0').astype(float)
    if 'Change' in df.columns:
        df['Change'] = df['Change'].astype(str).replace(['-', '#VALUE!'], '0').astype(float)
        
    # 2. Parse Date (Matching your notebook's '%d-%b-%y' format like '2-Jan-26')
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    except ValueError:
        # Fallback if the date format varies slightly
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        
    # 3. Drop rows missing critical raw data
    df = df.dropna(subset=['Day Price', 'Volume', '12m High', '12m Low'])
    
    # 4. Sort chronologically (CRITICAL for rolling windows)
    df = df.sort_values(by=['Date', 'Code']).reset_index(drop=True)
    return df

def engineer_features(df):
    """Computes the 10 required features from cleaned NSE stock data."""
    
    # 1. Volatility
    df['Vol_5'] = df.groupby('Code')['Day Price'].transform(lambda x: x.rolling(5, min_periods=2).std())
    df['Vol_10'] = df.groupby('Code')['Day Price'].transform(lambda x: x.rolling(10, min_periods=2).std())
    
    # 2. Price Ratios
    df['Price_to_High'] = df['Day Price'] / df['12m High']
    df['Price_to_Low'] = df['Day Price'] / df['12m Low']
    
    # 3. Daily Return
    df['Daily_Return'] = df.groupby('Code')['Day Price'].pct_change()
    
    # 4. Volume Features
    df['Volume_MA_5'] = df.groupby('Code')['Volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    
    # 5. RSI (14-day) - Using exact notebook logic to prevent Pandas warnings
    def calc_rsi(group, window=14):
        delta = group['Day Price'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        group['RSI_14'] = 100 - (100 / (1 + rs))
        return group
    df = df.groupby('Code', group_keys=False).apply(calc_rsi)
    
    # 6. MACD Histogram
    ema_12 = df.groupby('Code')['Day Price'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby('Code')['Day Price'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df.groupby('Code')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 7. Momentum (10-day)
    df['Momentum_10'] = df.groupby('Code')['Day Price'].pct_change(10)
    
    # Extract the latest row for each stock to predict today's movement
    latest_data = df.sort_values('Date').groupby('Code').tail(1)
    
    return latest_data[['Code', 'Date'] + expected_features]

# ==========================================
# RUN BATCH PREDICTION
# ==========================================
if __name__ == '__main__':
    # 1. Load your raw data
    print("Loading raw data...")
    # Ensure this filename matches your actual raw CSV file
    raw_data = pd.read_csv('NSE_data_all_stocks_2026_upto_jun.csv') 
    
    # 2. Clean data (Matching Notebook)
    print("Cleaning raw data formats...")
    cleaned_data = clean_raw_data(raw_data)
    
    # 3. Engineer features
    print("Calculating technical indicators...")
    features_df = engineer_features(cleaned_data)
    
    # 4. Drop NaNs (Stocks that didn't have enough history to form the 10 features)
    features_df = features_df.dropna(subset=expected_features)
    
    if features_df.empty:
        print("❌ No stocks had sufficient historical data to generate all 10 features.")
        exit(1)
    
    # 5. Predict
    print("Generating predictions...\n")
    predictions = model.predict(features_df[expected_features])
    probabilities = model.predict_proba(features_df[expected_features])[:, 1]
    
    # 6. Format Output
    features_df['Prediction'] = predictions
    features_df['Confidence_Up'] = (probabilities * 100).round(2)
    features_df['Signal'] = features_df['Prediction'].map({1: 'BUY / HOLD', 0: 'SELL / AVOID'})
    
    # Show results
    final_results = features_df[['Code', 'Date', 'Signal', 'Confidence_Up']]
    print(final_results.to_string(index=False))
    
    # Save to CSV
    final_results.to_csv('finsight_predictions_today.csv', index=False)
    print("\n✅ Predictions saved to finsight_predictions_today.csv")
