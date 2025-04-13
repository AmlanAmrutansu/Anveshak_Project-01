import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath, parse_dates=["Date"])
    data.sort_values("Date", inplace=True)
    data["Target"] = data["Close"].shift(-1)
    data = data[:-1]
    
    X = data[["Open", "High", "Low", "Close", "Volume"]].values
    y = data["Target"].values
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error on test data:", mse)
    
    return model

def main():
    X, y = load_and_prepare_data("stock_data.csv")
    model = train_model(X, y)
    joblib.dump(model, "stock_model.pkl")
    print("Model saved as stock_model.pkl")

if __name__ == "__main__":
    main()
