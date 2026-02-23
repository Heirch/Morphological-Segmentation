import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from data_loader import MorphDataset, collate_fn, SOS_TOKEN, PAD_TOKEN
from model import Encoder, Decoder, Attention, Seq2Seq
import time

# --- CONFIGURATION ---
BATCH_SIZE = 256
HIDDEN_DIM = 256
EMBED_DIM = 128
DROPOUT = 0.5
LEARNING_RATE = 0.0001  # <--- Lower learning rate for fine-tuning
EPOCHS = 20             # <--- 20 more epochs
CLIP = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Resuming training on: {device}")

# 1. LOAD DATA
dataset = MorphDataset("eng.train") 
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
vocab_size = dataset.vocab.n_chars

# 2. INITIALIZE MODEL
attn = Attention(HIDDEN_DIM)
enc = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT)
dec = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

# 3. LOAD PREVIOUS WEIGHTS
print("Loading previous best model...")
model.load_state_dict(torch.load('best-model.pt'))

# 4. OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.char2index[PAD_TOKEN])
scaler = GradScaler()

# 5. TRAINING LOOP
best_valid_loss = 0.14 # Approximate loss where we left off

print(f"Fine-tuning for {EPOCHS} more epochs...")

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(dataloader):
        src = src.to(device).permute(1, 0)
        trg = trg.to(device).permute(1, 0)

        optimizer.zero_grad()
        
        with autocast():
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(dataloader)
    end_time = time.time()
    
    # Save if better
    if avg_loss < best_valid_loss:
        best_valid_loss = avg_loss
        torch.save(model.state_dict(), 'best-model.pt')
        saved_msg = "- Saved Best Model!"
    else:
        saved_msg = ""

    print(f"Fine-Tune Epoch {epoch+1:02} | Loss: {avg_loss:.4f} | Time: {end_time - start_time:.1f}s {saved_msg}")