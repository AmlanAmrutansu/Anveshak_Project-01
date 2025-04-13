from flask import Flask, request, jsonify
from flask_cors import CORS
from ML import get_prediction  

app = Flask(__name__)
CORS(app)  

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to get stock market predictions.
    Expects a JSON payload with a "ticker" field.
    """
    try:
        data = request.get_json()
        if not data or "ticker" not in data:
            return jsonify({"error": "No ticker provided"}), 400
        
        ticker = data["ticker"]
        
        predictions, _, _, _, market_trend, message = get_prediction(ticker)

        if message:
            return jsonify({"error": message}), 400

        return jsonify({
            "predictions": predictions.tolist(),
            "market_trend": market_trend
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)  
