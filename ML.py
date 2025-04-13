import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import requests  # For potential integration with a real search API

app = Flask(__name__)
CORS(app)

# Configure logging with timestamps
logging.basicConfig(
    filename='ml_chat.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Load stock data from CSV at startup.
STOCK_DATA_CSV = 'stock_data.csv'
try:
    stock_df = pd.read_csv(STOCK_DATA_CSV, parse_dates=['Date'])
    stock_df.sort_values("Date", inplace=True)
except Exception as e:
    logging.error(f"Error loading stock data: {e}")
    stock_df = pd.DataFrame()

# In-memory conversation history for chat functionality.
chat_history = []

def ai_reply(message):
    """
    Process the input message and return a simulated AI response.
    """
    reply_text = f"I received your message: '{message}'. This is a simulated reply."
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return f"{reply_text} [at {timestamp} UTC]"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        logging.error("No message received in /chat request")
        return jsonify({'reply': "No message received."}), 400

    user_message = data['message']
    logging.info(f"User message: {user_message}")
    
    reply = ai_reply(user_message)
    logging.info(f"Generated reply: {reply}")
    
    conversation_entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_message,
        "ai": reply
    }
    chat_history.append(conversation_entry)
    
    return jsonify({'reply': reply})

@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    """
    Returns the conversation history.
    """
    return jsonify({"chat_history": chat_history})

@app.route('/reset_history', methods=['POST'])
def reset_history():
    """
    Reset the in-memory chat history.
    """
    chat_history.clear()
    logging.info("Chat history has been reset.")
    return jsonify({"message": "Chat history reset."}), 200

# StockHub endpoints

@app.route('/stocks', methods=['GET'])
def get_stocks():
    """
    Returns the stock data loaded from CSV as JSON.
    Optionally filters by exchange or ticker if query parameters are provided.
    Example: /stocks?exchange=NSE&ticker=RELIANCE
    """
    df = stock_df.copy()
    exchange = request.args.get('exchange')
    ticker = request.args.get('ticker')
    if exchange:
        df = df[df['Exchange'].str.upper() == exchange.upper()]
    if ticker:
        df = df[df['Ticker'].str.upper() == ticker.upper()]
    result = df.to_dict(orient='records')
    return jsonify({"stocks": result})

@app.route('/stock_chart/<ticker>', methods=['GET'])
def get_stock_chart(ticker):
    """
    Generates a line chart for the specified ticker using closing prices over time.
    Returns a base64 encoded PNG image in JSON.
    """
    df = stock_df[stock_df['Ticker'].str.upper() == ticker.upper()]
    if df.empty:
        return jsonify({"error": "Ticker not found."}), 404

    # Create the plot.
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df['Date'], df['Close'], marker='o', linestyle='-', label=ticker.upper())
    ax.set_title(f"{ticker.upper()} Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a bytes buffer.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return jsonify({"ticker": ticker.upper(), "chart": image_base64})

@app.route('/stock_search', methods=['POST'])
def stock_search():
    """
    Simulates an internet search for stock-related content.
    Accepts a JSON payload with a 'query' field.
    In a real implementation, integrate with Bing or another search API.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No search query provided."}), 400

    query = data['query']
    logging.info(f"Performing stock search for query: {query}")
    
    # Simulated search results. Replace this with real API calls if desired.
    simulated_results = [
        {"title": "Stock Market News", "snippet": "Latest updates on the stock market.", "url": "https://www.marketnews.com"},
        {"title": f"Analysis of {query.upper()} Stock", "snippet": "Detailed analysis and predictions.", "url": f"https://www.stockanalysis.com/{query.lower()}"},
        {"title": "Investing Basics", "snippet": "Learn the basics of investing in stocks.", "url": "https://www.investopedia.com"}
    ]
    return jsonify({"results": simulated_results})

if __name__ == '__main__':
    # Run on port 5000 to serve all endpoints.
    app.run(host='0.0.0.0', port=5000, debug=True)
