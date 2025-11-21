import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
from encodec import EncodecModel
import soundfile as sf
from tqdm import tqdm

# Use MPS on Apple Silicon, CUDA on NVIDIA, otherwise CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

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

# ---- Generation parameters ---------------------------------------------
TEMPERATURE = 0.9  # Higher = more random, lower = more deterministic
TOP_K = 50  # Only sample from top K tokens
TOP_P = 0.95  # Nucleus sampling threshold
NUM_TOKENS = 3000  # Generate ~20 seconds of audio

def sample_with_temperature(logits, temperature=1.0, top_k=50, top_p=0.95):
    """Sample from logits with temperature, top-k, and top-p filtering."""
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# ---- Generate tokens ---------------------------------------------------
print(f"Generating {NUM_TOKENS} tokens...")
seq = torch.randint(0, 1024, (1, 64), device=DEVICE)  # random start

for _ in tqdm(range(NUM_TOKENS), desc="Generating"):
    # Only use last 2048 tokens for context (model's max position)
    context = seq[:, -2048:] if seq.shape[1] > 2048 else seq
    logits = model(context).logits[:, -1, :]
    next_token = sample_with_temperature(logits, TEMPERATURE, TOP_K, TOP_P)
    seq = torch.cat([seq, next_token], dim=1)

tokens = seq[0].detach().cpu().numpy()  # shape [N]
print(f"Generated {len(tokens)} tokens")

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

# Create encoded frames in the format expected by decode: list of (codes, scale) tuples
# Scale can be None for the basic decoder
encoded_frames = [(tokens_tensor, None)]

with torch.no_grad():
    wav = codec.decode(encoded_frames)  # Returns [B, C, T] audio

# wav has shape [1, 1, T] (batch, channels, time)
wav_np = wav[0, 0].cpu().numpy()  # Extract to 1D array
sf.write("generated.wav", wav_np, 24000)
print("Saved generated.wav")
