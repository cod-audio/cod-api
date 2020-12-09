from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, Heroku!'

@app.route('/api/label-track', methods=['POST'])
def label_track():
    pass
