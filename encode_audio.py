import os
import torch
import torchaudio
from encodec import EncodecModel
from tqdm import tqdm
import numpy as np

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # smaller = fewer tokens, easier to train

DATA_DIR = "data/wavs/"
OUT_DIR = "data/tokens/"
os.makedirs(OUT_DIR, exist_ok=True)

def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    # Convert stereo to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # Average channels to mono
    wav = wav.unsqueeze(0)   # (1, channels, time)
    return wav

for file in tqdm(os.listdir(DATA_DIR)):
    if not file.endswith(".wav"):
        continue

    wav = load_audio(os.path.join(DATA_DIR, file))

    with torch.no_grad():
        encoded_frames = model.encode(wav)  # returns encoded frames tuple

    # encoded_frames is a tuple: (codes_tensor, None) or just codes_tensor
    # codes_tensor has shape [B, K, T] where B=batch, K=codebooks, T=timesteps
    if isinstance(encoded_frames, tuple):
        codes = encoded_frames[0][0].cpu().numpy()  # [B, K, T] -> [K, T]
    else:
        codes = encoded_frames[0].cpu().numpy()  # [B, K, T] -> [K, T]

    np.save(os.path.join(OUT_DIR, file.replace(".wav", ".npy")), codes)
