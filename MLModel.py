import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def load_and_prepare_data(filepath):
    # Load historical stock data; CSV must include at least the following columns: Date, Open, High, Low, Close, Volume.
    data = pd.read_csv(filepath, parse_dates=["Date"])
    data.sort_values("Date", inplace=True)
    
    # Create a new column 'Target' which is the next day's closing price.
    data["Target"] = data["Close"].shift(-1)
    # Drop the last row where target will be NaN.
    data = data[:-1]
    
    # Select features and target. Here we're using Open, High, Low, Close, and Volume as features.
    X = data[["Open", "High", "Low", "Close", "Volume"]].values
    y = data["Target"].values
    
    return X, y

def train_model(X, y):
    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Create and train the linear regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model performance on test set.
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error on test data:", mse)
    
    return model

def main():
    # Load and prepare data.
    X, y = load_and_prepare_data("stock_data.csv")
    
    # Train the model.
    model = train_model(X, y)
    
    # Save the model using joblib.
    joblib.dump(model, "stock_model.pkl")
    print("Model saved as stock_model.pkl")

if __name__ == "__main__":
    main()
