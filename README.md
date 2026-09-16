# Word2Vec Skip-Gram Architecture with Negative Sampling

An end-to-end PyTorch implementation of the Skip-Gram model with Negative Sampling (SGNS) based on the seminal paper [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013). 

This project implements a clean, modular NLP pipeline from scratch—handling vocabulary construction, context window pair generation, noise distribution sampling, embedding training, vector similarity evaluation, and high-dimensional visualization.

---

## 📌 Features

- **Modular Design**: Decoupled dataset logic (`dataset.py`), pre-processing utilities (`preprocessing.py`), network architecture (`model.py`), and experiment tracking (`experiments.ipynb`).
- **Skip-Gram Pair Generation**: Sliding-window extraction of (center, context) token target pairs.
- **Negative Sampling (SGNS)**: Custom implementation of unigram distribution raised to the $3/4$ power ($P_n(w) \propto f(w)^{0.75}$) to efficiently approximate softmax over large vocabularies.
- **Dual Embedding Architecture**: Dual `nn.Embedding` lookup tables for target center words and context/noise words with vectorized dot-product scoring.
- **Evaluation & Visualization**: Integrated cosine similarity nearest-neighbors analysis and t-SNE dimensionality reduction plots.

---

## 🛠 Project Structure

```text
word2vec-skipgram/
├── dataset.py          # PyTorch Dataset & context-window pair generator
├── preprocessing.py    # Vocabulary builder & unigram noise distribution loader
├── model.py            # Word2VecSkipGram PyTorch module & custom loss function
├── experiments.ipynb   # End-to-end training, similarity queries & t-SNE plots
└── README.md           # Project documentation
```

---

## 📐 Mathematical Overview

### 1. Skip-Gram Objective
The objective of the Skip-Gram model is to find word representations that are useful for predicting surrounding words in a sentence or document. Given a sequence of training words $w_1, w_2, \dots, w_T$, the objective is to maximize the average log probability:

$$\mathcal{L}_{SG} = rac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j 
eq 0} \log P(w_{t+j} \mid w_t)$$

where $c$ is the training context window size.

### 2. Negative Sampling (SGNS) Loss
Computing standard Softmax requires summing over the entire vocabulary $V$, which is computationally expensive $O(|V|)$:

$$P(w_O \mid w_I) = rac{\exp({v'_{w_O}}^	op v_{w_I})}{\sum_{w=1}^{|V|} \exp({v'_w}^	op v_{w_I})}$$

Mikolov et al. replace Softmax with Negative Sampling, turning the multi-class classification task into $k+1$ binary logistic regression tasks:

$$\mathcal{L}_{SGNS} = -\log \sigma({v'_{w_O}}^	op v_{w_I}) - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-{v'_{w_i}}^	op v_{w_I}) 
\right]$$

where:
- $v_w$ and $v'_w$ are input (center) and output (context) vector representations of $w$.
- $\sigma(x) = rac{1}{1 + e^{-x}}$ is the sigmoid activation function.
- $k$ is the number of drawn negative samples per positive target pair.
- $P_n(w) \propto f(w)^{0.75}$ is the unigram noise distribution penalizing overly frequent stopwords while giving rare words a higher probability of being sampled.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with PyTorch, NLTK, Scikit-learn, and Matplotlib:

```bash
pip install torch nltk scikit-learn matplotlib
```

### Quickstart

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mahdimeskin/word2vec-skipgram
   cd word2vec-skipgram
   ```

2. **Run Experiments**:
   Launch Jupyter Notebook or Jupyter Lab to run the pipeline end-to-end:
   ```bash
   jupyter notebook experiments.ipynb
   ```

---

## 💡 Code Overview

### Vocabulary & Preprocessing (`preprocessing.py`)
Builds frequency-filtered dictionaries (`word_to_idx`, `idx_to_word`) and generates the smoothed noise probability tensor:

```python
from collections import Counter
import torch

def build_vocab(tokens, min_count=2):
    word_counts = Counter(tokens)
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

def create_negative_sampling_distribution(tokens, word_to_idx):
    word_counts = Counter(tokens)
    frequencies = [word_counts[word] for word, _ in sorted(word_to_idx.items(), key=lambda x: x[1])]
    weights = torch.tensor(frequencies, dtype=torch.float32) ** 0.75
    return weights / weights.sum()
```

### PyTorch Model (`model.py`)
Computes positive pair dot-products and batch matrix multiplication (`torch.bmm`) for $k$ negative candidates:

```python
import torch
import torch.nn as nn

class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, center_ids, context_ids, negative_ids):
        center_vectors = self.input_embeddings(center_ids)           # [batch_size, embed_dim]
        positive_vectors = self.output_embeddings(context_ids)       # [batch_size, embed_dim]
        negative_vectors = self.output_embeddings(negative_ids)      # [batch_size, num_neg, embed_dim]
        
        positive_scores = torch.sum(center_vectors * positive_vectors, dim=1)
        negative_scores = torch.bmm(negative_vectors, center_vectors.unsqueeze(2)).squeeze(2)
        
        return positive_scores, negative_scores

    def loss(self, positive_scores, negative_scores):
        loss_fn = nn.BCEWithLogitsLoss()
        pos_loss = loss_fn(positive_scores, torch.ones_like(positive_scores))
        neg_loss = loss_fn(negative_scores, torch.zeros_like(negative_scores))
        return pos_loss + neg_loss
```

---

## 📊 Results & Visualization

### Word Similarity Queries (Cosine Distance)
```python
# Nearest neighbors extracted using trained embedding weights
Similar to 'alice': ['she', 'thought', 'said', 'little', 'went']
Similar to 'queen': ['king', 'hatter', 'duchess', 'shouted', 'said']
```

### Embedding Space Projection (t-SNE)
The notebook generates a 2D t-SNE plot demonstrating semantic clustering (e.g., characters, verbs, and dialogue words clustering together):

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Dimensionality reduction from 50D -> 2D space
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
vectors_2d = tsne.fit_transform(embeddings)
```

---

## 📜 References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781).
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NIPS 2013.
