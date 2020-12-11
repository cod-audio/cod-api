from flask import Flask, request
import numpy as np

from typing import List
from predict import IALModel, MODEL_PATH, CLASSLIST_PATH

app = Flask(__name__)
model = IALModel(MODEL_PATH, CLASSLIST_PATH)

@app.route('/')
def hello_world():
    return '<h1>Hello, Heroku!</h1>'

@app.route('/api/label-track', methods=['POST'])
def label_track():
    audio: List = list(request.form['buffer'])
    sr: float = request.form['sampleRate']

    audio_array: np.ndarray = np.asarray(audio)

    # Force it into a 2D array
    if len(audio_array.shape) == 1:
        audio_array = np.reshape(audio_array, (1, -1))

    labels: List[str] = model.predict_from_audio_array(audio_array, sr)

    ret = {'labels':[]}

    last_label = ''
    for time, label in enumerate(labels):
        if label != last_label:
            ret['labels'].append({'label':label, 'start':time})

    return ret
