import os

import yaml
import numpy as np
import torch
import librosa

import utils

MODEL_PATH = 'assets/model.pt'
CLASSLIST_PATH = 'assets/classlist.yaml'

class IALModel:

    def init(self, path_to_model: str, path_to_classlist: str, 
            ):
        """ loads a torch.jit model for inference in audio. 
        """
        self.model = load_jit_model(path_to_model)
        self.model.eval()

        # TODO: it would be good to check that the length of the classlist 
        # matched the output of the model
        self.classlist = utils.load_classlist(path_to_classlist)

        # some constants
        self.conf_threshold = 0.3
        self.uncertain_class = 'uncertain'
        self.sample_rate = 48000

    def predict_from_audio_array(self, audio: np.ndarray, sample_rate: int) -> list[str]:
        """predict musical instrument classes from an audio array

        Args:
            audio (np.ndarray): audio array. must be shape (channels, time)
            sample_rate (int): input sample rate

        Returns:
            list[str]: list of class probabilities for each frame
        """
        utils.__check_audio_types(audio)
        # resample and downmix if needed
        audio = resample(audio, sr, self.sample_rate)
        audio = downmix(audio)
        audio = zero_pad(audio)

        # DEBUG
        utils.__check_audio_types(audio)

        # convert to torch tensor!
        audio = torch.from_numpy(audio)

        # reshape to batch dimension
        audio = audio.view(-1, 1, self.sample_rate)

        # get class probabilities from model 
        probabilities = self.model(audio)
        
        # get the prediction indices by getting the argmax 
        prediction_indices = torch.argmax(probabilities, dim=1)
        confidences = torch.amax(probabilities, dim=1)

        # get list of predictions for every second
        prediction_classes = [self.classlist[idx] for conf, idx in zip(confidences, prediction_indices) 
                                if conf > self.conf_threshold else self.uncertain_class]
        
        return prediction_class

    def predict_from_audio_file(self, path_to_audio) -> list[str]:
        audio = utils.load_audio_file(path_to_audio, sample_rate=self.sample_rate)
        return self.predict_from_audio_array(audio, self.sample_rate)


if __name__ == "__main__":
    model = IALModel(path_to_model=MODEL_PATH, path_to_classlist=CLASSLIST_PATH)

    # load test audio
    audio = utils.load_audio_file('assets/drum-set.wav',model.sample_rate)

    print(predictions)
