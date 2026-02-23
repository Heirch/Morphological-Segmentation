import torch
from model import Encoder, Decoder, Attention, Seq2Seq
from data_loader import MorphDataset, SOS_TOKEN

# Configuration matches your training script
HIDDEN_DIM = 256
EMBED_DIM = 128
DROPOUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    print("Loading Polyglot Dictionary...")
    # Load dataset to get the vocabulary (characters)
    dataset = MorphDataset("multi_indic.train")
    vocab_size = dataset.vocab.n_chars
    
    print(f"Loading Model (Vocab Size: {vocab_size})...")
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT)
    dec = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    # Load the trained weights
    model.load_state_dict(torch.load('indic-model.pt', map_location=DEVICE))
    model.eval()
    return model, dataset

def predict(model, dataset, word, lang_tag):
    # Prepend the tag: "khaata" -> "<HIN>khaata"
    tagged_word = f"<{lang_tag}>{word}"
    
    src_indices = dataset.vocab.word_to_indices(tagged_word, add_eos=True)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        trg_indices = [dataset.vocab.char2index[SOS_TOKEN]]
        
        for i in range(30):
            trg_tensor = torch.LongTensor([trg_indices[-1]]).to(DEVICE)
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
            pred_token = output.argmax(1).item()
            if pred_token == dataset.vocab.char2index['<EOS>']:
                break
            trg_indices.append(pred_token)
            
    return dataset.vocab.indices_to_word(trg_indices)

if __name__ == "__main__":
    model, dataset = load_model()
    
    # List of languages you trained on
    langs = ['ENG', 'HIN', 'BEN', 'KAN', 'TEL', 'URD']
    
    print("\n" + "="*40)
    print("   INDIC POLYGLOT MODEL READY")
    print("="*40)
    
    while True:
        print(f"\nSupported: {langs}")
        lang = input("Select Language Code (or 'q'): ").strip().upper()
        if lang == 'Q': break
        
        if lang not in langs:
            print(f"Unknown language. Please use one of: {langs}")
            continue
            
        word = input(f"Enter {lang} word: ").strip()
        if not word: continue
        
        result = predict(model, dataset, word, lang)
        print(f"Result: {word} -> {result}")




