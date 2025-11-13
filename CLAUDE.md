# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a music generation project that uses a GPT-2 transformer model to learn and generate music by training on audio encoded as discrete tokens via Meta's Encodec codec. The approach is similar to language models but applied to audio.

## Architecture

The project consists of three main components in a sequential pipeline:

1. **Audio Encoding** (`encode_audio.py`): Converts raw audio files to discrete tokens
   - Uses Encodec (24kHz, 6.0 bandwidth) to tokenize audio
   - Reads WAV files from `data/wavs/`
   - Outputs token arrays (.npy) to `data/tokens/`
   - Handles stereo-to-mono conversion and resampling to 24kHz
   - Encodec produces multiple codebooks; currently only the first codebook is used

2. **Model Training** (`train.py`): Trains GPT-2 on audio tokens
   - Loads tokenized audio from `data/tokens/`
   - Uses a small GPT-2 (256 embd, 6 layers, 8 heads, 2048 context)
   - Vocabulary size: 1024 (matches Encodec codebook size)
   - Trains for 10 epochs with batch size 2
   - Saves model weights to `mini_suno.pth`
   - Uses only the first codebook from Encodec for simplicity

3. **Audio Generation** (`generate.py`): Generates new audio from trained model
   - Loads the trained model from `mini_suno.pth`
   - Starts with random tokens and autoregressively generates ~500 new tokens
   - Uses argmax sampling (greedy decoding)
   - Decodes generated tokens back to audio using Encodec
   - Outputs to `generated.wav` at 24kHz

## Data Flow

```
WAV files (data/wavs/)
  → encode_audio.py →
NPY tokens (data/tokens/)
  → train.py →
Model checkpoint (mini_suno.pth)
  → generate.py →
Generated audio (generated.wav)
```

## Development Commands

**Setup:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Training Pipeline:**
```bash
# 1. Encode audio files to tokens
python encode_audio.py

# 2. Train the model
python train.py

# 3. Generate new audio
python generate.py
```

**Requirements:**
- PyTorch with CUDA support (optional, will fall back to CPU)
- Place training audio (.wav files) in `data/wavs/` before encoding
- Requires at least ~1GB of audio data for meaningful results

## Key Technical Details

- **Device Selection**: All scripts automatically detect and use CUDA if available, otherwise fall back to CPU (defined as `DEVICE` constant)
- **Codebook Simplification**: Encodec produces multiple codebooks, but this implementation uses only the first codebook (`arr[0]`) to simplify training
- **Padding**: Training uses dynamic padding in the collate function to handle variable-length sequences
- **Token Vocabulary**: Fixed at 1024 tokens to match Encodec's codebook size
- **Audio Format**: All audio is converted to 24kHz mono during encoding

## Important Model Configuration

The GPT-2 config in both `train.py` and `generate.py` must match exactly:
- vocab_size: 1024
- n_positions: 2048
- n_embd: 256
- n_layer: 6
- n_head: 8

If you modify the architecture in one file, update both files accordingly.
