from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configure logging with timestamps
logging.basicConfig(
    filename='ml_chat.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# In-memory conversation history (for demonstration purposes)
chat_history = []

def ai_reply(message):
    """
    Process the input message and return an AI response.
    Replace this dummy logic with your ML-based chat functionality.
    Extra features include:
      - A timestamped reply
      - Logging for debugging purposes
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
    
    # Generate AI reply
    reply = ai_reply(user_message)
    logging.info(f"Generated reply: {reply}")
    
    # Save the conversation to history with timestamp
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
    Endpoint to retrieve the entire chat conversation history.
    """
    return jsonify({"chat_history": chat_history})

@app.route('/reset_history', methods=['POST'])
def reset_history():
    """
    Endpoint to clear the in-memory chat history.
    """
    chat_history.clear()
    logging.info("Chat history has been reset.")
    return jsonify({"message": "Chat history reset."}), 200

if __name__ == '__main__':
    # Start the Flask app to serve the AI chat endpoint
    app.run(host='0.0.0.0', port=5000, debug=True)
