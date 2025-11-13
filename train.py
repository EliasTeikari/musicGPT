import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, AdamW
from tqdm import tqdm

TOK_DIR = "data/tokens/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset -------------------------------------------------------------
class TokenDataset(Dataset):
    def __init__(self):
        self.files = [f for f in os.listdir(TOK_DIR) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(os.path.join(TOK_DIR, self.files[idx]), allow_pickle=True)
        # arr has shape [K, T] (codebooks, timesteps) → choose first codebook
        tokens = arr[0].astype(np.int64).flatten()
        return torch.tensor(tokens)

def collate(batch):
    # Pad sequences to same length
    lens = [len(x) for x in batch]
    max_len = max(lens)
    padded = [torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in batch]
    return torch.stack(padded), torch.stack(padded)

dataset = TokenDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

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

optimizer = AdamW(model.parameters(), lr=3e-4)

# --- Training Loop -------------------------------------------------------
for epoch in range(10):
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x, labels=y)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": loss.item()})

torch.save(model.state_dict(), "mini_suno.pth")
print("Saved model → mini_suno.pth")
