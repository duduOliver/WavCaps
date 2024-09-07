import torch
import torch.nn as nn
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
import torch.nn.functional as F
import copy
from tools.losses import AudioTextContrastiveLoss, NTXent
from tools.utils import remove_grad
import ruamel.yaml as yaml
import librosa
import random
import numpy as np
from models.feature_extractor import DACodes
from time import sleep

config_file_path = './retrieval/settings/dac.yaml'

with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
        
# Load audio signal file
wav_file_path = '../dac/audio_samples/at2_cvt.wav'
waveform, _ = librosa.load(wav_file_path, sr=config["audio_args"]["sr"], mono=True)
print('waveform shape before crop: ', waveform.shape)
if config["audio_args"]["max_length"] != 0:
            # if audio length is longer than max_length, we random crop it
            max_length = config["audio_args"]["max_length"] * config["audio_args"]["sr"]
            if waveform.shape[-1] > max_length:
                max_start = waveform.shape[-1] - max_length
                start = random.randint(0, max_start)
                waveform = waveform[start: start + max_length]
                

                
print('waveform shape: ', waveform.shape)
waveform_tensor = torch.tensor(waveform[None, :])
print('waveform_tensor shape: ', waveform_tensor.shape)

batch_size = 16
batch_waveform_tensor = waveform_tensor.repeat(batch_size, 1)
print("batch_waveform_tensor.shape: ", batch_waveform_tensor.shape)

# wav_file_path = '../dac/audio_samples/at1_cvt.wav'
# waveform, _ = librosa.load(wav_file_path, sr=config["audio_args"]["sr"], mono=True)
# print('waveform shape before crop: ', waveform.shape)
# if config["audio_args"]["max_length"] != 0:
#             # if audio length is longer than max_length, we random crop it
#             max_length = config["audio_args"]["max_length"] * config["audio_args"]["sr"]
#             if waveform.shape[-1] > max_length:
#                 max_start = waveform.shape[-1] - max_length
#                 start = random.randint(0, max_start)
#                 waveform = waveform[start: start + max_length]
# batch_waveform_tensor[4] = torch.tensor(waveform)


for _ in range(5):
    AF = DACodes(config)
    audio_encoded = AF(batch_waveform_tensor.to("cuda"))
    print('audio_encoded.shape: ', audio_encoded.shape)
    sleep(3)

