import os
import logging
logging.basicConfig(format='IALMODEL:%(asctime)s:%(message)s', level=logging.WARNING)

import yaml
import numpy as np
import torch
import librosa

import utils

MODEL_PATH = 'assets/model.pt'
CLASSLIST_PATH = 'assets/classlist.yaml'

class IALModel:

    def __init__(self, path_to_model: str, path_to_classlist: str):
        """ loads a torch.jit model for inference in audio. 
        """
        self.model = utils.load_jit_model(path_to_model)
        self.model.eval()

        # TODO: it would be good to check that the length of the classlist 
        # matched the output of the model
        self.classlist = utils.load_classlist(path_to_classlist)

        # some constants
        self.conf_threshold = 0.3
        self.uncertain_class = 'uncertain'
        self.sample_rate = 48000

    def predict_from_audio_array(self, audio: np.ndarray, sample_rate: int):
        """predict musical instrument classes from an audio array

        Args:
            audio (np.ndarray): audio array. must be shape (channels, time)
            sample_rate (int): input sample rate

        Returns:
            list[str]: list of class probabilities for each frame
        """
        logging.info(f'got audio with shape {audio.shape}')
        utils._check_audio_types(audio)
        # resample, downmix, and zero pad if needed
        logging.info('resampling...')
        audio = utils.resample(audio, sample_rate, self.sample_rate)

        logging.info('downmixing...')
        audio = utils.downmix(audio)

        logging.info('zero padding...')
        audio = utils.zero_pad(audio)

        # convert to torch tensor!
        audio = torch.from_numpy(audio)

        # reshape to batch dimension
        # TODO: need to enforce a maximum batch size 
        # to avoid OOM errors
        audio = audio.view(-1, 1, self.sample_rate)

        # get class probabilities from model 
        logging.info(f'got a batch size of {audio.shape[0]}. doing forward pass...')
        probabilities = self.model(audio)
        logging.info(f'done!')
        
        # get the prediction indices by getting the argmax 
        prediction_indices = torch.argmax(probabilities, dim=1)
        confidences = torch.amax(probabilities, dim=1)

        # get list of predictions for every second
        prediction_classes = [self.classlist[idx] if conf > self.conf_threshold else self.uncertain_class
                                for conf, idx in zip(confidences, prediction_indices)]
        logging.info(f'predictions: {prediction_classes}')
        
        return prediction_classes

    def predict_from_audio_file(self, path_to_audio):
        logging.info(f'received audio path {path_to_audio}. loading...')
        audio = utils.load_audio_file(path_to_audio, sample_rate=self.sample_rate)
        return self.predict_from_audio_array(audio, self.sample_rate)

def _quick_model_test():
    model = IALModel(path_to_model=MODEL_PATH, path_to_classlist=CLASSLIST_PATH)

    # load test audio
    audio = utils.load_audio_file('assets/electric-bass.wav',model.sample_rate)
    predictions = model.predict_from_audio_array(audio, model.sample_rate)
    assert predictions[0] == 'electric bass', predictions

    # or do it directly with a file path
    predictions = model.predict_from_audio_file('assets/drum-set.wav')
    assert predictions[0] == 'drum set', predictions

if __name__ == "__main__":
    model = IALModel(path_to_model=MODEL_PATH, path_to_classlist=CLASSLIST_PATH)
     # load test audio
    audio = utils.load_audio_file('/Users/hugo/Downloads/sarasate_op20_ff_01_moderato.mov',model.sample_rate)
    predictions = model.predict_from_audio_array(audio, model.sample_rate)