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

    # 1. Handle string anomalies in numeric columns
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    if 'Change%' in df.columns:
        df['Change%'] = df['Change%'].astype(str).str.replace('%', '', regex=False) \
            .replace(['-', '#VALUE!'], '0').astype(float)

    if 'Change' in df.columns:
        df['Change'] = df['Change'].astype(str) \
            .replace(['-', '#VALUE!'], '0').astype(float)

    # 2. Parse Date
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    except ValueError:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # 3. Drop rows missing critical raw data
    df = df.dropna(subset=['Day Price', 'Volume', '12m High', '12m Low'])

    # 4. Sort chronologically
    df = df.sort_values(by=['Date', 'Code']).reset_index(drop=True)
    return df


def engineer_features(df):
    """Computes the 10 required features from cleaned NSE stock data."""
   # 1. Lag features (past prices)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Lag_{lag}'] = df.groupby('Code')['Day Price'].shift(lag)

    # 2. Rolling averages and volatility
    for window in [5, 10, 20]:
        df[f'MA_{window}'] = df.groupby('Code')['Day Price'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'Vol_{window}'] = df.groupby('Code')['Day Price'].transform(
            lambda x: x.rolling(window, min_periods=2).std())

    # 3. Price ratios & Returns
    df['Price_to_High'] = df['Day Price'] / df['12m High']
    df['Price_to_Low'] = df['Day Price'] / df['12m Low']
    df['Daily_Return'] = df.groupby('Code')['Day Price'].pct_change()

    # 4. Date features
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter

    # 5. Volume features
    df['Volume_MA_5'] = df.groupby('Code')['Volume'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']

    # Advanced Technical Indicators
    # EMAs
    for window in [5, 10, 20]:
        df[f'EMA_{window}'] = df.groupby('Code')['Day Price'].transform(
            lambda x: x.ewm(span=window, adjust=False).mean())

    # RSI (14-day)
    def calc_rsi(group, window=14):
        delta = group['Day Price'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss= -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        group['RSI_14'] = 100 - (100 / (1 + rs))
        return group
    df = df.groupby('Code', group_keys=False).apply(calc_rsi)

    # MACD
    ema_12 = df.groupby('Code')['Day Price'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby('Code')['Day Price'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df.groupby('Code')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Momentum (10-day)
    df['Momentum_10'] = df.groupby('Code')['Day Price'].pct_change(10)

    # Extract the latest row for each stock
    latest_data = df.sort_values('Date').groupby('Code').tail(1)
    return latest_data[['Code', 'Date'] + expected_features]



# Run the prediction

if __name__ == '__main__':
    print("Loading raw data...")
    raw_data = pd.read_csv('NSE_data_all_stocks_2026_upto_jun.csv')

    print("Cleaning raw data formats...")
    cleaned_data = clean_raw_data(raw_data)

    print("Calculating technical indicators...")
    features_df = engineer_features(cleaned_data)

    features_df = features_df.dropna(subset=expected_features)

    if features_df.empty:
        print("❌ No stocks had sufficient historical data.")
        exit(1)

    print("Generating predictions...\n")
    predictions = model.predict(features_df[expected_features])
    probabilities = model.predict_proba(features_df[expected_features])[:, 1]

    features_df['Prediction'] = predictions
    features_df['Confidence_Up'] = (probabilities * 100).round(2)
    features_df['Signal'] = features_df['Prediction'] \
        .map({1: 'BUY / HOLD', 0: 'SELL / AVOID'})

    final_results = features_df[['Code', 'Date', 'Signal', 'Confidence_Up']]
    print(final_results.to_string(index=False))

    final_results.to_csv('finsight_predictions_today.csv', index=False)
    print("\n✅ Predictions saved to finsight_predictions_today.csv")
