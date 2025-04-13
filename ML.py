from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configure logging to file with timestamp and message details
logging.basicConfig(
    filename='ml_chat.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# In-memory store for chat history per session (for demonstration)
chat_history = []

def ai_reply(message):
    """
    Process the input message and produce an AI bot response.
    Replace or extend this dummy logic with your actual ML-based chat logic.
    Extra features include:
      - Timestamps added to replies
      - Custom processing could be integrated here
    """
    # Here, you may call your ML model or any additional processing.
    # For demonstration, we simply echo the message with a custom reply.
    reply_text = f"I received your message: '{message}'. This is a simulated reply."
    # Optionally, add a timestamp to the reply.
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return f"{reply_text} [at {timestamp} UTC]"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        logging.error("No message received in request")
        return jsonify({'reply': "No message received."}), 400

    user_message = data['message']
    logging.info(f"User message received: {user_message}")
    
    # Generate a reply via your ML logic
    reply = ai_reply(user_message)
    logging.info(f"Reply generated: {reply}")

    # Save the conversation to history along with timestamp
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
    Endpoint to retrieve the chat conversation history.
    This can be extended to filter or paginate history as needed.
    """
    return jsonify({"chat_history": chat_history})

@app.route('/reset_history', methods=['POST'])
def reset_history():
    """
    Reset the in-memory chat history.
    """
    chat_history.clear()
    logging.info("Chat history reset by user request.")
    return jsonify({"message": "Chat history has been reset."}), 200

if __name__ == '__main__':
    # Run on port 5000 to serve the AI chat endpoint.
    app.run(host='0.0.0.0', port=5000, debug=True)
