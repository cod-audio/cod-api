from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return '<h1>Hello, Heroku!</h1>'

@app.route('/api/label-track', methods=['POST'])
def label_track():
    pass
