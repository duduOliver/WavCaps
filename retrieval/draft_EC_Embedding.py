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
from models.encodec_embedding import EncodecEmbedding

config_file_path = './retrieval/settings/vamp.yaml'

with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

EcEmbed = EncodecEmbedding(device="cuda")
        
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

batch_size = 128
batch_waveform_tensor = waveform_tensor.repeat(batch_size, 1)
print("batch_waveform_tensor.shape: ", batch_waveform_tensor.shape)

wav_file_path = '../dac/audio_samples/at1_cvt.wav'
waveform, _ = librosa.load(wav_file_path, sr=EcEmbed.sampling_rate, mono=True)
print('waveform shape before crop: ', waveform.shape)
if config["audio_args"]["max_length"] != 0:
            # if audio length is longer than max_length, we random crop it
            max_length = config["audio_args"]["max_length"] * config["audio_args"]["sr"]
            if waveform.shape[-1] > max_length:
                max_start = waveform.shape[-1] - max_length
                start = random.randint(0, max_start)
                waveform = waveform[start: start + max_length]
batch_waveform_tensor[1] = torch.tensor(waveform)

# print("batch_waveform_tensor[0] and batch_waveform_tensor[1] sim: ", torch.mean((batch_waveform_tensor[0] - batch_waveform_tensor[1]) ** 2))
# # for _ in range(10):
# em_0 = EcEmbed.embedding(batch_waveform_tensor[0].repeat(2, 1))
# print("em_0 shape:", em_0.shape)

# em_1 = EcEmbed.embedding(batch_waveform_tensor[1].repeat(2, 1))
# print("em_1 shape:", em_1.shape)
# print("The similarity  is: ", torch.mean((em_0.float() - em_1.float()) ** 2))

batch_waveform_tensor_np = batch_waveform_tensor[:,None,:].repeat(1, 2, 1)#.detach().cpu().numpy().tolist()
print("batch_waveform_tensor[:,None,:].repeat(1,2,1) shape:", batch_waveform_tensor_np.shape)
# print( bool( isinstance(batch_waveform_tensor_np, (list, tuple)) and (isinstance(batch_waveform_tensor_np[0], (np.ndarray, tuple, list))) ))
# EcEmbed.model.eval()
with torch.no_grad():
    em_bt = EcEmbed.embedding(batch_waveform_tensor_np)
# em_bt = [
#     EcEmbed.embedding(bw)[None,:] for bw in batch_waveform_tensor_np
# ] 

# c_vamp = torch.cat(em_bt)
# print("c_vamp shape:", c_vamp.shape)
em_bt_ft = torch.flatten(em_bt.permute(1, 0, 2, 3), start_dim=1)
print("em_bt shape: ", em_bt_ft.shape)
print("em_bt_ft[0] and em_bt_ft[1] sim: ", torch.mean((em_bt_ft[0] - em_bt_ft[1]) ** 2))
print("em_bt_ft[0] and em_bt_ft[2] sim: ", torch.mean((em_bt_ft[0] - em_bt_ft[2]) ** 2))

