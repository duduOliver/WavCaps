from pathlib import Path
import audiotools as at #a helpful library from Descript for dealing with wavefiles
import vampnet
import torch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,3,4,5"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# print(Path(__file__).parent.parent.parent)
# print("#########")
# import sys
# os.path.split(os.getcwd())[0]
# sys.path.append("/gpfs/home/dhuang/thesis/WavCaps/retrieval/models")
# sys.path.append("/gpfs/home/dhuang/thesis/WavCaps/retrieval/tools")
# for nb_dir in sys.path:
    # print(nb_dir)
    # sys.path.append(nb_dir)

# load the audio tokenizer. The tokenizer takes audio waveforms and turns them into
# a low bitrate stream of tokens that make it easier for a Transformer (like vampnet)
# to process
codec = vampnet.load_codec()

# load the default pretrained model
model = vampnet.load_model(vampnet.DEFAULT_MODEL)

# An Interface is how you interact with vampnet to control it. Let's put the codec and
# model into an interface
interface = vampnet.interface.Interface(codec, model)

# THis line moves our processing to the GPU, which is much faster than using the CPU.
# NOTE: In Google colab, go to the "Runtime/Change-runtime-type" menu item and pick
# something that is not "CPU". This notebook was tested on colab with "T4 GPU" selected.
# interface = interface.to("cuda" if torch.cuda.is_available() else "cpu")

wav_file_path = 'dac/audio_samples/at2_16khz_cvt.wav' # from `thesis` path
signal = at.AudioSignal.excerpt(wav_file_path, duration=10.0)
# signal.widget()

# Get the tokens for the audio signal.
tokens = interface.encode(signal)

print('the input signal has shape: ', signal.shape, ' at sample-rate: ', signal.sample_rate)
print('the tokenized signal has shape: ', tokens.shape)

# Detokenizing the signal can be done easily, as follows.
signal_detokenized = interface.decode(tokens)
print('the detokenized signal has shape: ', signal_detokenized.shape, ' at sample-rate: ', signal_detokenized.sample_rate)

# Note that it decodes at a 44.1 kHz sample rate when detokenizing.
# Let's resample back up to 48 kHz
signal_detokenized = signal_detokenized.resample(48000)
print('the resampled signal has shape: ', signal_detokenized.shape, ' at sample-rate: ', signal_detokenized.sample_rate)

#pick which mask we'd like to use by uncommenting the appropriate line
#mask = mask_prefix
#mask = mask_suffix
mask = torch.ones_like(tokens)
# print(mask)
#mask = mask_details
#mask = mask_multi

# generate the output tokens from `x = self.embedding(x)` 1102080 = 1280*861
output_tokens = interface.vamp_em(
    tokens, mask, return_mask=False,
    sampling_temperature=1.0,
    typical_filtering=True,
    top_p=0.9,
    sample_cutoff=1.0,
)

print(output_tokens.shape)
print(output_tokens)
