import numpy as np
import librosa
import yaml
import warnings
import torch

def _check_audio_types(audio: np.ndarray):
    assert isinstance(audio, np.ndarray), f'expected np.ndarray but got {type(audio)} as input.'
    assert audio.ndim == 2, f'audio must be shape (channels, time), got shape {audio.shape}'
    if audio.shape[-1] < audio.shape[-2]:
        warnings.warn(f'IALMODELWARNING: got audio shape {audio.shape}. Audio should be (channels, time). \
                        typically, the number of samples is much larger than the number of channels. ')

def _is_mono(audio: np.ndarray):
    _check_audio_types(audio)
    num_channels = audio.shape[-2]
    return num_channels == 1


def load_audio_file(path_to_audio, sample_rate=48000):
    """ wrapper for loading mono audio with librosa
    returns:
        audio (np.ndarray): monophonic audio with shape (channels, samples) 
    """
    audio, sr = librosa.load(path_to_audio, mono=True, sr=sample_rate)
    # add channel dimension
    audio = np.expand_dims(audio, axis=-2)
    return audio

def load_jit_model(path_to_model: str) -> torch.jit.ScriptModule:
    """ loads a torch.jit model

    Args:
        path_to_model (str): path to the torch.jit model.
    """
    return torch.jit.load(path_to_model)

def load_classlist(path_to_classlist: str) -> list:
    """[summary]

    Args:
        path_to_classlist (str): path to classlist file in YAML format. 

    Returns:
        list: list of strings with classnames. 
    """
    with open(path_to_classlist, 'r') as f:
        classlist = yaml.load(f, Loader=yaml.SafeLoader)
    return classlist

def downmix(audio: np.ndarray):
    """ downmix an audio array. 
    must be shape (channels, mono)

    Args:
        audio ([np.ndarray]): array to downmix
    """
    _check_audio_types(audio)
    audio = audio.mean(axis=-2, keepdims=True)
    return audio

def resample(audio: np.ndarray, old_sr: int, new_sr: int = 48000) -> np.ndarray:
    """wrapper around librosa for resampling

    Args:
        audio (np.ndarray): audio array shape (channels, time)
        old_sr (int): old sample rate
        new_sr (int, optional): target sample rate.  Defaults to 48000.

    Returns:
        np.darray: resampled audio. shape (channels, time)
    """
    _check_audio_types(audio)

    if _is_mono(audio):
        audio = audio[0]
        audio = librosa.resample(audio, old_sr, new_sr)
        audio = np.expand_dims(audio, axis=-2)
    else:
        audio = librosa.resample(audio, old_sr, new_sr)
    return audio

def zero_pad(audio: np.ndarray, required_len: int = 48000) -> np.ndarray:
    """zero pad audio array to meet a multiple of required_len

    Args:
        audio (np.ndarray): audio array w shape (channels, sample)
        required_len (int, optional): target length in samples. Defaults to 48000.

    Returns:
        np.ndarray: zero padded audio
    """
    _check_audio_types(audio)

    num_frames = audio.shape[-1]

    before = 0
    after = required_len - num_frames%required_len
    audio = np.pad(audio, pad_width=((0, 0), (before, after)), mode='constant', constant_values=0)
    return audio
