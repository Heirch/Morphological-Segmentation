import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # <--- IMPORT FOR SPEED
from data_loader import MorphDataset, collate_fn, SOS_TOKEN, PAD_TOKEN
from model import Encoder, Decoder, Attention, Seq2Seq
import time
import os

# --- CONFIGURATION ---
# 256 is the sweet spot for 4GB VRAM with Mixed Precision
BATCH_SIZE = 256        
HIDDEN_DIM = 256
EMBED_DIM = 128
DROPOUT = 0.5
LEARNING_RATE = 0.001 
EPOCHS = 30
CLIP = 1.0

# CHECK FOR GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# 1. LOAD DATA (Real Dataset)
dataset = MorphDataset("eng.train") 
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

vocab_size = dataset.vocab.n_chars
print(f"Vocabulary Size: {vocab_size}")

# 2. INITIALIZE MODEL
attn = Attention(HIDDEN_DIM)
enc = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT)
dec = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)

# 3. OPTIMIZER & LOSS
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.char2index[PAD_TOKEN])
scaler = GradScaler() # <--- INIT SCALER FOR MIXED PRECISION

# 4. TRAINING LOOP
best_valid_loss = float('inf')

print(f"Starting training on {len(dataset)} words...")
print(f"Batch Size: {BATCH_SIZE} | Precision: Mixed (16-bit)")

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(dataloader):
        src = src.to(device).permute(1, 0) # [Len, Batch]
        trg = trg.to(device).permute(1, 0) # [Len, Batch]

        optimizer.zero_grad()
        
        # <--- AUTOCAST: The Magic Speed Trick
        with autocast():
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
        
        # Scale the loss and step (Required for Mixed Precision)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
        # Print progress every 50 batches
        if i % 50 == 0:
            print(f"\rEpoch {epoch+1} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}", end="")
        
    avg_loss = epoch_loss / len(dataloader)
    end_time = time.time()
    
    print(f"\rEpoch {epoch+1:02} | Loss: {avg_loss:.4f} | Time: {end_time - start_time:.1f}s    ")
    
    if avg_loss < best_valid_loss:
        best_valid_loss = avg_loss
        torch.save(model.state_dict(), 'best-model.pt')

# 5. TESTING PREDICTION
print("\n--- LOADING BEST MODEL FOR TESTING ---")
model.load_state_dict(torch.load('best-model.pt'))
model.eval()

def predict_word(word):
    src_indices = dataset.vocab.word_to_indices(word, add_eos=True)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        trg_indices = [dataset.vocab.char2index[SOS_TOKEN]]
        
        for i in range(30):
            trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
            pred_token = output.argmax(1).item()
            if pred_token == dataset.vocab.char2index['<EOS>']:
                break
            trg_indices.append(pred_token)
    
    return dataset.vocab.indices_to_word(trg_indices)

test_words = ["wolves", "running", "unhappiness", "baked", "morphology"]
for w in test_words:
    print(f"Input: {w:15} -> Predicted: {predict_word(w)}")