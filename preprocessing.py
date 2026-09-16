from collections import Counter
import torch

def build_vocab(tokens, min_count=1):
    word_counts = Counter(tokens)
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

def create_negative_sampling_distribution(tokens, word_to_idx):
    word_counts = Counter(tokens)
    frequencies = [
        word_counts[word] 
        for word, _ in sorted(word_to_idx.items(), key=lambda x: x[1])
    ]
    weights = torch.tensor(frequencies, dtype=torch.float32) ** 0.75
    return weights / weights.sum()