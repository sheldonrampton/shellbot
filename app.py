"""
shellbot_flask.py:
Runs a Flask-powered chatbot that answers questions using embeddings
extracted from Sheldon Rampton's social media and emails.

To run:
flask --app shellbot_flask run
"""

from flask import Flask, request, jsonify, session, render_template
from flask_session import Session  # Import Flask-Session
from openai import OpenAI # for calling the OpenAI API
from social_data import SocialData
from chatbotter import Asker, ConversationLogger
from flask_cors import CORS
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# In-memory storage for session data and logs
session_data = {}
user_logs = []

openai_client = OpenAI(
  organization=os.getenv('OPENAI_ORGANIZATION'),
  project=os.getenv('OPENAI_PROJECT'),
)
sd = SocialData(
    openai_client,
    knowledge_db_name = 'shellbot_knowledge',
    pinecone_index_name = "shellbot-embeddings2",
)
sd.setup_pinecone()

asker = Asker(openai_client, storage = sd,
    introduction = 'Use the below messages which were written by Sheldon Rampton to answer questions as though you are Sheldon Rampton. If the answer cannot be found in the articles, write "I could not find an answer."',
    string_divider = 'Messages:'
)
logger = ConversationLogger()

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)
app.secret_key = os.urandom(24)  # Secret key for session management
app.config['SESSION_TYPE'] = 'filesystem'  # You can also use 'redis', 'mongodb', etc.
app.config['SESSION_PERMANENT'] = False  # Session won't be permanent
Session(app)

@app.route("/")
def home():
    if 'session_id' not in session:
        session.clear()  # This clears the session data on the server
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
        session_data[session_id] = []  # Initialize conversation history
        print(f"New session initialized: {session_id}")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Check if the session ID exists
    session_id = session.get('session_id')
    
    if not session_id or session_id not in session_data:
        # If no session ID or session ID is not in the session_data, reinitialize
        session.clear()  # This clears the session data on the server
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
        session_data[session_id] = []  # Initialize conversation history
        print(f"New session initialized: {session_id}")
    
    user_input = request.json.get('message')
    # Retrieve conversation history
    conversation_history = session_data[session_id]
    bot_response, references, articles = asker.ask(user_input, conversation_history = conversation_history)
    if bot_response == "I could not find an answer.":
        session.clear()  # This clears the session data on the server
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
        session_data[session_id] = []  # Initialize conversation history
        print(f"New session initialized: {session_id}")
        bot_response, references, articles = asker.ask(user_input, conversation_history = conversation_history)

    conversation_history.append({"role": "user", "content": user_input})

    # Append bot response to the conversation history
    conversation_history.append({"role": "assistant", "content": bot_response})

    # Log the conversation (if necessary)
    log_entry = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "bot_response": bot_response
    }
    user_logs.append(log_entry)
    logger.post_entry(log_entry)

    return jsonify({"response": bot_response})

@app.route('/get_logs', methods=['GET'])
def get_logs():
    return jsonify(user_logs)

@app.route('/list_logs', methods=['GET'])
def list_logs():
    logs = logger.get_entries()
    return jsonify(logs)


if __name__ == '__main__':
    app.run(debug=True)
