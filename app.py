from flask import Flask, request, jsonify
from ML import get_prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get("ticker")

    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    tomorrow, next_month, low, high, trend, message = get_prediction(ticker)

    if message:
        return jsonify({"error": message}), 400

    return jsonify({
        "tomorrow_open": tomorrow[0],
        "tomorrow_close": tomorrow[1],
        "next_month": next_month.tolist(),
        "confidence_range": [low, high],
        "market_trend": trend
    })

if __name__ == "__main__":
    app.run(debug=True)
