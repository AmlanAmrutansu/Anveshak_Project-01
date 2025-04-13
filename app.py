from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing if your frontend is hosted separately

def get_prediction(ticker):
    """
    Simulate a stock prediction function.
    In a real scenario, this would use a machine learning model or data processing algorithm.
    """
    # Dummy data for demonstration purposes.
    try:
        ticker = ticker.upper()
        result = {
            "market_trend": "Bullish" if ticker[0] < "N" else "Bearish",
            "tomorrow_open": 142.35,
            "tomorrow_close": 145.67,
            "next_month": [150.00, 152.30, 149.80],
            "confidence_range": [140.00, 155.00],
            "error": None
        }
    except Exception as e:
        result = { "error": f"Failed to process prediction: {str(e)}" }
    return result

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "ticker" not in data:
        return jsonify({"error": "No ticker provided"}), 400
    ticker = data["ticker"]
    prediction = get_prediction(ticker)
    if prediction.get("error"):
        return jsonify({"error": prediction["error"]}), 500
    return jsonify(prediction)

if __name__ == "__main__":
    # Bind to PORT if defined otherwise use 5000.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
