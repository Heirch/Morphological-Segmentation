import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from data_loader import MorphDataset, collate_fn, SOS_TOKEN, PAD_TOKEN
from model import Encoder, Decoder, Attention, Seq2Seq
import time
import sys

# --- SMART CONFIGURATION ---
# Batch Size 256 is the "Sweet Spot" for your RTX 3050 (Speed vs Memory)
BATCH_SIZE = 256
HIDDEN_DIM = 256
EMBED_DIM = 128
DROPOUT = 0.5
LEARNING_RATE = 0.001
EPOCHS = 20  # 15 Epochs on this massive dataset is equal to 50 on a small one
CLIP = 1.0

# 1. SETUP DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training Polyglot Indic Model on: {device}")

# 2. LOAD DATA
dataset_file = "multi_indic.train"
print(f"Loading Dataset: {dataset_file}...")

try:
    dataset = MorphDataset(dataset_file)
except FileNotFoundError:
    print(f"ERROR: {dataset_file} not found. Did you run prepare_indic_polyglot.py?")
    sys.exit()

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

vocab_size = dataset.vocab.n_chars
print(f"Total Vocabulary Size: {vocab_size}")
print(f"Total Training Samples: {len(dataset)}")

# 3. INITIALIZE SMART MODEL
attn = Attention(HIDDEN_DIM)
enc = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT)
dec = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

# Initialize weights nicely (Helps it learn faster)
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
model.apply(init_weights)

# 4. OPTIMIZER & LOSS
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.char2index[PAD_TOKEN])
scaler = GradScaler() # The "Speed" Scaler

# 5. TRAINING LOOP
best_valid_loss = float('inf')

print("\n--- STARTING INTELLIGENT TRAINING ---")
print(f"Estimated batches per epoch: {len(dataloader)}")

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(dataloader):
        src = src.to(device).permute(1, 0)
        trg = trg.to(device).permute(1, 0)

        optimizer.zero_grad()
        
        # Mixed Precision Context (Speed Mode)
        with autocast():
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
        
        # Gradient Scaling (Stability Mode)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
        # Progress Bar (Every 50 batches)
        if i % 50 == 0:
             print(f"\rEpoch {epoch+1} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}", end="")
        
    avg_loss = epoch_loss / len(dataloader)
    end_time = time.time()
    
    # Save Model Logic
    if avg_loss < best_valid_loss:
        best_valid_loss = avg_loss
        torch.save(model.state_dict(), 'indic-model.pt')
        saved_msg = "- Saved Best Brain!"
    else:
        saved_msg = ""

    print(f"\rEpoch {epoch+1:02} | Loss: {avg_loss:.4f} | Time: {end_time - start_time:.1f}s {saved_msg}    ")

print("\nTraining Complete! The polyglot brain is ready: 'indic-model.pt'")