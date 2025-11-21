import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

TOK_DIR = "data/tokens/"
# Use MPS on Apple Silicon, CUDA on NVIDIA, otherwise CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# --- Dataset -------------------------------------------------------------
MAX_SEQ_LEN = 2048  # Must match model's n_positions

class TokenDataset(Dataset):
    def __init__(self):
        self.chunks = []
        # Load all files and create overlapping chunks for more training data
        for f in os.listdir(TOK_DIR):
            if not f.endswith(".npy"):
                continue
            arr = np.load(os.path.join(TOK_DIR, f), allow_pickle=True)
            tokens = arr[0].astype(np.int64).flatten()

            # Create multiple chunks from each file with stride
            stride = MAX_SEQ_LEN // 2  # 50% overlap
            for i in range(0, max(1, len(tokens) - MAX_SEQ_LEN + 1), stride):
                chunk = tokens[i:i + MAX_SEQ_LEN]
                if len(chunk) >= 512:  # Only use chunks with enough data
                    self.chunks.append(chunk)

        print(f"Created {len(self.chunks)} training chunks from {len([f for f in os.listdir(TOK_DIR) if f.endswith('.npy')])} files")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return torch.tensor(chunk, dtype=torch.long)

def collate(batch):
    # Pad sequences to same length
    lens = [len(x) for x in batch]
    max_len = max(lens)
    padded = [torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in batch]
    return torch.stack(padded), torch.stack(padded)

dataset = TokenDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)

# --- Model ---------------------------------------------------------------
VOCAB_SIZE = 1024      # Encodec has ~1024 tokens per codebook
config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=2048,
    n_embd=256,
    n_layer=6,
    n_head=8
)
model = GPT2LMHeadModel(config).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# --- Training Loop -------------------------------------------------------
NUM_EPOCHS = 100
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    num_batches = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x, labels=y)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "mini_suno.pth")

print(f"Training complete. Best loss: {best_loss:.4f}")
print("Saved model → mini_suno.pth")
