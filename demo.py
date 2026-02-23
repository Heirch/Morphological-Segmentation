import torch
from model import Encoder, Decoder, Attention, Seq2Seq
from data_loader import MorphDataset, SOS_TOKEN

# --- CONFIGURATION ---
HIDDEN_DIM = 256
EMBED_DIM = 128
DROPOUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    print("Loading Dictionary...")
    # We load the dataset just to get the Vocabulary (char -> ID mapping)
    dataset = MorphDataset("eng.train")
    vocab_size = dataset.vocab.n_chars
    
    print(f"Loading Model (Vocab Size: {vocab_size})...")
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT)
    dec = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    # Load the weights we just trained
    model.load_state_dict(torch.load('best-model.pt', map_location=DEVICE))
    model.eval()
    return model, dataset

def predict(model, dataset, word):
    # 1. Convert text to numbers
    src_indices = dataset.vocab.word_to_indices(word, add_eos=True)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        
        # Start with <SOS>
        trg_indices = [dataset.vocab.char2index[SOS_TOKEN]]
        
        # Generate characters one by one
        for i in range(30):
            trg_tensor = torch.LongTensor([trg_indices[-1]]).to(DEVICE)
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
            
            # Pick the letter with highest probability
            pred_token = output.argmax(1).item()
            
            if pred_token == dataset.vocab.char2index['<EOS>']:
                break
            trg_indices.append(pred_token)
            
    # 2. Convert numbers back to text
    return dataset.vocab.indices_to_word(trg_indices)

# --- MAIN LOOP ---
if __name__ == "__main__":
    model, dataset = load_model()
    print("\n" + "="*40)
    print("MORPHOLOGICAL SEGMENTATION AI READY")
    print("Type 'q' to quit.")
    print("="*40 + "\n")

    while True:
        word = input("Enter a word: ").strip().lower()
        if word == 'q': break
        
        if len(word) > 0:
            result = predict(model, dataset, word)
            print(f"Result: {word} -> {result}\n")