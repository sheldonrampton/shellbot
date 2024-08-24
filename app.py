"""
shellbot_flask.py:
Runs a Flask-powered chatbot that answers questions using embeddings
extracted from Sheldon Rampton's social media and emails.

To run:
flask --app shellbot_flask run
"""

from flask import Flask, request, jsonify, render_template
from openai import OpenAI # for calling the OpenAI API
from social_data import SocialData
from chatbotter import Asker
from flask_cors import CORS


openai_client = OpenAI(
  organization='***REMOVED***',
  project='***REMOVED***',
)
sd = SocialData(openai_client)
sd.setup_pinecone()

asker = Asker(openai_client, storage = sd,
    introduction = 'Use the below messages which were written by Sheldon Rampton to answer questions as though you are Sheldon Rampton. If the answer cannot be found in the articles, write "I could not find an answer."',
    string_divider = 'Messages:'
)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get('message')
    response, references, articles = asker.ask(query)
    # print(references)
    # print(articles)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
