import torch
from torch.utils.data import Dataset

class SkipGramDataset(Dataset):
    def __init__(self, pairs, word_to_idx):
        self.indexed_pairs = [
            (word_to_idx[center], word_to_idx[context]) 
            for center, context in pairs
        ]
    
    def __len__(self):
        return len(self.indexed_pairs)
    
    def __getitem__(self, idx):
        center, context = self.indexed_pairs[idx]
        return (
            torch.tensor(center, dtype=torch.long), 
            torch.tensor(context, dtype=torch.long)
        )

def generate_skipgram_pairs(tokens, window_size=2):
    pairs = []
    for center_idx, center_word in enumerate(tokens):
        start = max(0, center_idx - window_size)
        end = min(len(tokens), center_idx + window_size + 1)
        for context_idx in range(start, end):
            if context_idx != center_idx:
                pairs.append((center_word, tokens[context_idx]))
    return pairs