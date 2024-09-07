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
from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch

config_file_path = './retrieval/settings/vamp.yaml'

with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
        
max_length = config["audio_args"]["max_length"] * config["audio_args"]["sr"]
        
# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
print("target_bandwidths have:", model.target_bandwidths) # [1.5, 3.0, 6, 12.0, 24.0] 
model.set_target_bandwidth(6.0) # means 8 quantizers, 24.0 == 32 n_q

wav, sr = torchaudio.load("../dac/audio_samples/at2_cvt.wav")
if wav.shape[-1] > max_length:
    max_start = wav.shape[-1] - max_length
    start = random.randint(0, max_start)
    wav = wav[:, start: start + max_length]
wav = wav.unsqueeze(0)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)


print("wav shape:", wav.shape)


        
# # Load audio signal file
# wav_file_path = '../dac/audio_samples/at2_cvt.wav'
# waveform, _ = librosa.load(wav_file_path, sr=config["audio_args"]["sr"], mono=True)
# print('waveform shape before crop: ', waveform.shape)
# if config["audio_args"]["max_length"] != 0:
#             # if audio length is longer than max_length, we random crop it
#             max_length = config["audio_args"]["max_length"] * config["audio_args"]["sr"]
#             if waveform.shape[-1] > max_length:
#                 max_start = waveform.shape[-1] - max_length
#                 start = random.randint(0, max_start)
#                 waveform = waveform[start: start + max_length]
                

                
# print('waveform shape: ', wav.shape)
# waveform_tensor = torch.tensor(wav[None, :])
# print('waveform_tensor shape: ', waveform_tensor.shape)

batch_size = 32
batch_waveform_tensor = wav.repeat(batch_size, 1, 1)
print("batch_waveform_tensor.shape: ", batch_waveform_tensor.shape)

wav, sr = torchaudio.load("../dac/audio_samples/at1_cvt.wav")
if wav.shape[-1] > max_length:
    max_start = wav.shape[-1] - max_length
    start = random.randint(0, max_start)
    wav = wav[:, start: start + max_length]
wav = wav.unsqueeze(0)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
print("wav shape:", wav.shape)
batch_waveform_tensor[1] = wav



print("batch_waveform_tensor[0] and batch_waveform_tensor[1] sim: ", torch.mean((batch_waveform_tensor[0] - batch_waveform_tensor[4]) ** 2))
# for _ in range(10):
encoded_frames = model.encode(batch_waveform_tensor[0:1,:,:])
codes_0 = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
print("codes_0 shape:", codes_0.shape)
audio_embeds_0_0 = model.quantizer.vq.layers[0]._codebook.embed
# print(audio_feats)
# print('model.quantizer.vq.layers: ', model.quantizer.vq.layers)
# print("model.quantizer.vq.layers[0]: ", model.quantizer.vq.layers[0])
# print("model.quantizer.vq.layers[0]._codebook: ", model.quantizer.vq.layers[0]._codebook)
print("The similarity of 0-1 layers in a same forward is: ", torch.mean((model.quantizer.vq.layers[0]._codebook.embed - model.quantizer.vq.layers[1]._codebook.embed) ** 2))

encoded_frames = model.encode(batch_waveform_tensor[1:2,:,:])
codes_1 = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
print("codes_1 shape:", codes_1.shape)
audio_embeds_1_0 = model.quantizer.vq.layers[0]._codebook.embed
print("The similarity of 0 layer in different forward is: ", torch.mean((codes_0.float() - codes_1.float()) ** 2))
print("The similarity of 0 layer in different forward is: ", torch.mean((audio_embeds_0_0 - audio_embeds_1_0) ** 2))
