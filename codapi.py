from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np

from typing import List
from predict import IALModel, MODEL_PATH, CLASSLIST_PATH

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def hello_world():
    return '<h1>Hello, Heroku!</h1>'

@app.route('/api/label-track', methods=['POST'])
@cross_origin()
def label_track():
    response.headers.add('Access-Control-Allow-Origin', '*')

    audio: List = list(request.json['buffer'].values())
    del request.json['buffer']

    sr: float = request.json['sampleRate']
    model = IALModel(MODEL_PATH, CLASSLIST_PATH)

    audio_array: np.ndarray = np.asarray(audio, dtype=np.float32)
    del audio

    # Force it into a 2D array
    if len(audio_array.shape) == 1:
        audio_array = np.reshape(audio_array, (1, -1))

    labels: List[str] = model.predict_from_audio_array(audio_array, sr)

    ret = {'labels':[]}

    last_label = ''
    for time, label in enumerate(labels):
        if label != last_label:
            ret['labels'].append({'label':label, 'start':time})
            last_label = label

    return ret
