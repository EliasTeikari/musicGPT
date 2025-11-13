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
    wav = wav.unsqueeze(0)   # (1, channels, time)
    return wav

for file in tqdm(os.listdir(DATA_DIR)):
    if not file.endswith(".wav"):
        continue

    wav = load_audio(os.path.join(DATA_DIR, file))

    with torch.no_grad():
        encoded = model.encode(wav)[0]   # list of tensors, one per codebook

    # Convert to numpy and save
    codes = [e.cpu().numpy() for e in encoded]
    np.save(os.path.join(OUT_DIR, file.replace(".wav", ".npy")), codes)
