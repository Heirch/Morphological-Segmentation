import torch
from model import Encoder, Decoder, Attention, Seq2Seq
from data_loader import MorphDataset, SOS_TOKEN

# 1. SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 256
EMBED_DIM = 128
DROPOUT = 0.5

# 2. LOAD DATA (Just for vocabulary)
dataset = MorphDataset("eng.train")
vocab_size = dataset.vocab.n_chars

# 3. INITIALIZE MODEL (Empty Brain)
attn = Attention(HIDDEN_DIM)
enc = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT)
dec = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

# 4. LOAD WEIGHTS (Fill Brain with Knowledge)
print("Loading model...")
model.load_state_dict(torch.load('best-model.pt'))
model.eval() # Set to evaluation mode

# 5. PREDICTION FUNCTION
def segment_word(word):
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

# 6. INTERACTIVE LOOP
while True:
    user_input = input("\nEnter a word to segment (or 'q' to quit): ")
    if user_input.lower() == 'q': break
    print(f"Segmentation: {segment_word(user_input)}")