import torch
from torch.utils.data import Dataset, DataLoader
import os

# --- CONSTANTS ---
PAD_TOKEN = '<PAD>' # Padding
SOS_TOKEN = '<SOS>' # Start of Sequence
EOS_TOKEN = '<EOS>' # End of Sequence
UNK_TOKEN = '<UNK>' # Unknown

class MorphVocabulary:
    def __init__(self):
        self.char2index = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.index2char = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.n_chars = 4

    def add_word(self, word):
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.n_chars += 1

    def word_to_indices(self, word, add_eos=False):
        indices = [self.char2index.get(char, self.char2index[UNK_TOKEN]) for char in word]
        if add_eos:
            indices.append(self.char2index[EOS_TOKEN])
        return indices

    def indices_to_word(self, indices):
        # Convert list of numbers back to string, ignoring special tokens
        return "".join([self.index2char.get(idx, UNK_TOKEN) for idx in indices 
                        if idx not in [0, 1, 2]])

class MorphDataset(Dataset):
    def __init__(self, filepath, vocab=None):
        self.pairs = []
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at {filepath}")

        # 1. Load Data
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                
                input_word = parts[0]
                # Clean target: "un @@happy" -> "unhappy" (or keep markers if you prefer)
                # For this specific project, we usually keep them or replace with +
                target_word = parts[1].replace(' @@', '+') 
                
                self.pairs.append((input_word, target_word))
        
        # 2. Build Vocabulary (only if not provided)
        if vocab is None:
            self.vocab = MorphVocabulary()
            for src, trg in self.pairs:
                self.vocab.add_word(src)
                self.vocab.add_word(trg)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_word, trg_word = self.pairs[idx]
        
        # Convert text to numbers
        src_indices = self.vocab.word_to_indices(src_word, add_eos=True)
        
        # CRITICAL FIX: Add SOS token to the START of the target
        trg_raw = self.vocab.word_to_indices(trg_word, add_eos=True)
        trg_indices = [self.vocab.char2index[SOS_TOKEN]] + trg_raw
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

def collate_fn(batch):
    """
    This function pads the batches so all words have the same length.
    """
    src_batch, trg_batch = zip(*batch)
    
    # Pad with 0 (PAD_TOKEN)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=0, batch_first=True)
    
    return src_padded, trg_padded

# --- TEST BLOCK ---
# This only runs if you run this file directly.
if __name__ == "__main__":
    print("--- STARTING DATA LOADER TEST ---")
    
    # 1. Create a dummy file
    with open("dummy_train.txt", "w", encoding="utf-8") as f:
        f.write("wolves\twolf @@s\n")
        f.write("running\trun @@ing\n")
        f.write("baked\tbake @@ed\n")
    print("Created dummy_train.txt")

    # 2. Load it
    dataset = MorphDataset("dummy_train.txt")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    print(f"Vocabulary Size: {dataset.vocab.n_chars}")
    print(f"Char 'w' is index: {dataset.vocab.char2index.get('w')}")

    # 3. Check a Batch
    src_batch, trg_batch = next(iter(dataloader))
    
    print("\nBatch Output Shapes (Should be [2, Length]):")
    print(f"Source: {src_batch.shape}")
    print(f"Target: {trg_batch.shape}")
    
    print("\nDecoded First Example in Batch:")
    print("Input:", dataset.vocab.indices_to_word(src_batch[0].tolist()))
    print("Target:", dataset.vocab.indices_to_word(trg_batch[0].tolist()))
    
    print("\n--- TEST PASSED ---")