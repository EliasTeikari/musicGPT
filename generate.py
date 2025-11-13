import torch
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
from encodec import EncodecModel
import soundfile as sf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer/vocoder
codec = EncodecModel.encodec_model_24khz().to(DEVICE)
codec.set_target_bandwidth(6.0)

# Load model
config = GPT2Config(
    vocab_size=1024,
    n_positions=2048,
    n_embd=256,
    n_layer=6,
    n_head=8
)
model = GPT2LMHeadModel(config).to(DEVICE)
model.load_state_dict(torch.load("mini_suno.pth", map_location=DEVICE))
model.eval()

# ---- Generate tokens ---------------------------------------------------
seq = torch.randint(0, 1024, (1, 32), device=DEVICE)  # random start

for _ in range(500):  # generate ~500 future tokens
    logits = model(seq)
    next_token = torch.argmax(logits.logits[:, -1], dim=-1, keepdim=True)
    seq = torch.cat([seq, next_token], dim=1)

tokens = seq[0].detach().cpu().numpy()  # shape [N]

# ---- Decode tokens back to audio ---------------------------------------
# Encodec needs all codebooks [B, K, T], but we only trained on codebook 0
# Get number of codebooks from the model (6.0 kbps uses 8 codebooks)
num_codebooks = codec.quantizer.n_q

# Create tensor with shape [1, num_codebooks, T]
# Put generated tokens in first codebook, pad rest with zeros
T = len(tokens)
codes = np.zeros((1, num_codebooks, T), dtype=np.int64)
codes[0, 0, :] = tokens  # Fill first codebook with generated tokens

tokens_tensor = torch.tensor(codes).to(DEVICE)

with torch.no_grad():
    wav = codec.decode(tokens_tensor)  # Returns [B, C, T] audio

# wav has shape [1, 1, T] (batch, channels, time)
wav_np = wav[0, 0].cpu().numpy()  # Extract to 1D array
sf.write("generated.wav", wav_np, 24000)
print("Saved generated.wav")
