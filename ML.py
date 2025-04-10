import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xgboost as xgb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from newspaper import Article
from transformers import pipeline
from datetime import datetime, timedelta
import holidays

def is_market_closed():
    india_holidays = holidays.India()
    tomorrow = datetime.today() + timedelta(days=1)
    if tomorrow.weekday() >= 5 or tomorrow in india_holidays:
        return True, f"🚨 Market is closed on {tomorrow.strftime('%Y-%m-%d')}. No predictions available."
    return False, None

def get_live_stock_price(ticker):
    stock = yf.Ticker(ticker)
    live_price = stock.history(period="1d")['Close'].values[-1]
    return live_price

def get_live_macro_data():
    url = "https://api.worldbank.org/v2/country/IND/indicator/NY.GDP.MKTP.CD?format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data[-1]['value']
    except Exception as e:
        print("Error fetching macroeconomic data:", e)
        return None

def get_vix():
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="5y")
    return vix_data[['Close']]

def fetch_news_sentiment(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        sentiment = sentiment_analyzer(article.text)
        return sentiment
    except Exception as e:
        print("Error fetching sentiment:", e)
        return None

def compute_technical_indicators(data):
    data['SMA_7'] = data['Close'].rolling(window=7).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['STD_20'] = data['Close'].rolling(window=20).std()
    data['Bollinger_Upper'] = data['SMA_20'] + (2 * data['STD_20'])
    data['Bollinger_Lower'] = data['SMA_20'] - (2 * data['STD_20'])
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR_14'] = true_range.rolling(window=14).mean()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    return data

def preprocess_data(data):
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_7", "RSI_14", "SMA_20", "Bollinger_Upper",
        "Bollinger_Lower", "MACD", "MACD_Signal", "ATR_14", "Log_Return"
    ]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[feature_cols].values)
    print(f"Data scaled shape before reshaping: {data_scaled.shape}")
    data_scaled = data_scaled.reshape(data_scaled.shape[0], 1, data_scaled.shape[1])
    return data_scaled, scaler

def prepare_tomorrow_prediction(data):
    data["Tomorrow_Open"] = data["Open"].shift(-1)
    data["Tomorrow_Close"] = data["Close"].shift(-1)
    data.dropna(inplace=True)
    return data

def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LSTM(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(25, activation='relu')(x)
    outputs = Dense(2)(x) 
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_rf_model(data_train, labels_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(data_train, labels_train)
    return rf_model

def build_xgb_model(data_train, labels_train):
    base_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model = MultiOutputRegressor(base_xgb)
    xgb_model.fit(data_train, labels_train)
    return xgb_model

def train_models(data_scaled, labels):
    X_train = data_scaled[:-1]   
    y_train = labels[:-1]

    print("X_train shape before LSTM:", X_train.shape)
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = build_lstm_model(input_shape)
    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1, callbacks=[early_stop])
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    rf_model = build_rf_model(X_train_flat, y_train)
    xgb_model = build_xgb_model(X_train_flat, y_train)
    
    return lstm_model, rf_model, xgb_model

def predict_tomorrow(lstm, rf, xgb_model, data_scaled):
    lstm_input = data_scaled[-1].reshape(1, 1, data_scaled.shape[2])
    lstm_pred = lstm.predict(lstm_input)
    
    rf_input = data_scaled[-1].reshape(1, data_scaled.shape[1] * data_scaled.shape[2])
    rf_pred = rf.predict(rf_input)
    xgb_pred = xgb_model.predict(rf_input)
    
    final_pred = (0.33 * lstm_pred.flatten() +
                  0.33 * rf_pred.flatten() +
                  0.34 * xgb_pred.flatten())
    return final_pred

def predict_next_month(lstm, rf, xgb_model, data_scaled):
    if data_scaled.shape[0] < 30:
        raise ValueError("Not enough data for next month prediction")
    
    lstm_input = data_scaled[-30:].reshape(1, 30, data_scaled.shape[2])
    lstm_pred = lstm.predict(lstm_input)
    
    rf_preds = []
    xgb_preds = []
    for sample in data_scaled[-30:]:
        sample_flat = sample.reshape(1, data_scaled.shape[1] * data_scaled.shape[2])
        rf_preds.append(rf.predict(sample_flat)[0])
        xgb_preds.append(xgb_model.predict(sample_flat)[0])
    rf_pred_avg = np.mean(rf_preds, axis=0)
    xgb_pred_avg = np.mean(xgb_preds, axis=0)
    
    final_pred = (0.33 * lstm_pred.flatten() +
                  0.33 * rf_pred_avg +
                  0.34 * xgb_pred_avg)
    return final_pred

def evaluate_models(processed_data, labels):
    split_index = int(len(processed_data) * 0.8)
    X_train, X_test = processed_data[:split_index], processed_data[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]
    
    lstm, rf, xgb_model = train_models(X_train, y_train)
    
    predictions = []
    for i in range(len(X_test)):
        sample = X_test[i].reshape(1, X_test.shape[1], X_test.shape[2])
        lstm_pred = lstm.predict(sample)
        sample_flat = X_test[i].reshape(1, -1)
        rf_pred = rf.predict(sample_flat)
        xgb_pred = xgb_model.predict(sample_flat)
        final_pred = (0.33 * lstm_pred.flatten() +
                      0.33 * rf_pred.flatten() +
                      0.34 * xgb_pred.flatten())
        predictions.append(final_pred)
    
    predictions = np.array(predictions)
    rmse_open = np.sqrt(mean_squared_error(y_test[:, 0], predictions[:, 0]))
    rmse_close = np.sqrt(mean_squared_error(y_test[:, 1], predictions[:, 1]))
    mape_open = mean_absolute_percentage_error(y_test[:, 0], predictions[:, 0])
    mape_close = mean_absolute_percentage_error(y_test[:, 1], predictions[:, 1])
    
    print("Evaluation Metrics on Test Set:")
    print(f"RMSE (Open): {rmse_open:.2f}, RMSE (Close): {rmse_close:.2f}")
    print(f"MAPE (Open): {mape_open * 100:.2f}%, MAPE (Close): {mape_close * 100:.2f}%")
    
    return predictions, (rmse_open, rmse_close, mape_open, mape_close)

def determine_market_trend(data):
    atr = data['ATR_14'].tail(20).mean()
    recent_changes = np.diff(data['Close'].tail(10).values)
    current_volatility = np.std(recent_changes)

    print("ATR (last 20 mean):", atr)
    print("Recent changes:", recent_changes)
    print("Current volatility (std):", current_volatility)
    print("Mean of recent changes:", np.mean(recent_changes))

    if current_volatility > 1.2 * atr:
        market_condition = "Volatile ⚡"
    elif current_volatility < 0.8 * atr:
        if np.mean(recent_changes) > 0:
            market_condition = "Bullish 📈"
        elif np.mean(recent_changes) < 0:
            market_condition = "Bearish 📉"
        else:
            market_condition = "Sideways ↔️"
    else:
        market_condition = "Sideways ↔️"

    return market_condition

def compute_confidence_interval(predictions):
    mean_pred = np.mean(predictions)
    std_dev = np.std(predictions)
    confidence_lower = mean_pred - (1.96 * std_dev)
    confidence_upper = mean_pred + (1.96 * std_dev)
    return confidence_lower, confidence_upper

def get_prediction(company_ticker):
    market_closed, message = is_market_closed()
    if market_closed:
        return None, None, None, None, None, message

    try:
        stock_data = yf.download(company_ticker, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    except Exception as e:
        return None, None, None, None, None, f"Error downloading data: {e}"
    if stock_data.empty:
        return None, None, None, None, None, "No stock data available."

    stock_data = compute_technical_indicators(stock_data)

    stock_data.dropna(inplace=True)
    
    market_trend = determine_market_trend(stock_data)
    
    stock_data = prepare_tomorrow_prediction(stock_data)
    
    processed_data, scaler = preprocess_data(stock_data)
    
    if processed_data.shape[0] < 31:
        return None, None, None, None, market_trend, "Insufficient data for training and predictions."
    
    labels = stock_data[["Tomorrow_Open", "Tomorrow_Close"]].values
    lstm, rf, xgb_model = train_models(processed_data, labels)
    tomorrow_predictions = predict_tomorrow(lstm, rf, xgb_model, processed_data)
    next_month_predictions = predict_next_month(lstm, rf, xgb_model, processed_data)
    conf_low, conf_high = compute_confidence_interval(tomorrow_predictions)
    
    return tomorrow_predictions, next_month_predictions, conf_low, conf_high, market_trend, None

if __name__ == "__main__":
    company = input("Enter stock ticker (e.g., TSLA, NVTK.ME, TCS.NS, SBIN.BO): ")
    
    tomorrow_predictions, next_month_predictions, conf_low, conf_high, market_trend, message = get_prediction(company)
    
    if message:
        print(f"\n🚨 {message}")
    else:
        print(f"\nMarket Condition for Next Month: **{market_trend}**")
        print(f"\nPredicted Opening Price for Tomorrow: {tomorrow_predictions[0]:.2f}")
        print(f"Predicted Closing Price for Tomorrow: {tomorrow_predictions[1]:.2f}")
        print(f"\nPredicted Closing Prices for Next Month (Aggregated): {next_month_predictions}")
        print(f"\nConfidence Interval for Tomorrow: {conf_low:.2f} - {conf_high:.2f}")
    
    
    run_evaluation = input("\nWould you like to run a backtesting evaluation? (yes/no): ").strip().lower()
    if run_evaluation in ['yes', 'y']:
        eval_data = yf.download(company, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        eval_data = compute_technical_indicators(eval_data)
        eval_data.dropna(inplace=True)
        eval_data = prepare_tomorrow_prediction(eval_data)
        processed_eval_data, _ = preprocess_data(eval_data)
        eval_labels = eval_data[["Tomorrow_Open", "Tomorrow_Close"]].values
        _, metrics = evaluate_models(processed_eval_data, eval_labels)
    
    print("\nTHANKS 💛")
