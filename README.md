# Word2Vec Skip-Gram with Negative Sampling

An end-to-end PyTorch implementation of the **Skip-Gram model with Negative Sampling (SGNS)** based on the seminal paper [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781) by Mikolov et al. (2013).

This project implements the core Word2Vec Skip-Gram training procedure from scratch, including vocabulary construction, context-window pair generation, negative sampling, embedding training, cosine-similarity evaluation, and t-SNE visualization.

The implementation is designed as an educational/research-oriented project to understand the mathematical and computational ideas behind Word2Vec rather than relying on a pre-built Word2Vec library.

---

## 📌 Features

* **Modular Design**: Separates dataset logic, preprocessing utilities, model architecture, and experiments.
* **Skip-Gram Pair Generation**: Generates `(center, context)` training pairs using a configurable context window.
* **Negative Sampling (SGNS)**: Samples negative words from a smoothed unigram distribution proportional to \(f(w)^{0.75}\).
* **Dual Embedding Architecture**: Uses separate input and output embedding matrices, following the Skip-Gram formulation.
* **Vectorized Training**: Computes positive and negative scores using PyTorch tensor operations.
* **Cosine Similarity**: Finds nearest neighbors in the learned embedding space.
* **t-SNE Visualization**: Projects learned embeddings into two dimensions for qualitative inspection.
* **Real Text Corpus**: The experiment uses *Alice's Adventures in Wonderland* from the NLTK Gutenberg corpus.

---

## 🛠 Project Structure

```text
word2vec-skipgram/
├── dataset.py          # PyTorch Dataset and Skip-Gram pair generation
├── preprocessing.py    # Vocabulary construction and negative-sampling distribution
├── model.py            # Word2Vec Skip-Gram model and loss function
├── experiments.ipynb   # Training, evaluation, and visualization
└── README.md           # Project documentation
```

---

## 📐 Mathematical Overview

### 1. Skip-Gram Objective

Given a sequence of training words

$$
w_1, w_2, \dots, w_T,
$$

the Skip-Gram model learns word representations by predicting surrounding context words from a center word.

The original Skip-Gram objective maximizes the average log probability of the context words:

$$
\mathcal{L}_{SG} =
\frac{1}{T}
\sum_{t=1}^{T}
\sum_{\substack{-c \leq j \leq c \\ j \neq 0}}
\log P(w_{t+j} \mid w_t),
$$

where \(c\) is the maximum context-window size.

In this implementation, a fixed context window is used to generate the training pairs.

---

### 2. Negative Sampling

Computing the full Softmax probability requires a sum over the entire vocabulary:

$$
P(w_O \mid w_I) =
\frac{
\exp({v'_{w_O}}^\top v_{w_I})
}{
\sum_{w=1}^{|V|}
\exp({v'_w}^\top v_{w_I})
}.
$$

For large vocabularies, this can be computationally expensive.

Negative Sampling replaces the full multiclass prediction problem with a binary classification objective. For a positive center-context pair $\((w_I, w_O)\)$, the model learns to distinguish the real context word from $\(k\)$ randomly sampled negative words.

The SGNS objective can be written as:

$$
\mathcal{L}_{SGNS} =
-\log \sigma({v'_{w_O}}^\top v_{w_I}) -
\sum_{i=1}^{k}
\mathbb{E}_{w_i \sim P_n(w)}
\left[
\log \sigma(-{v'_{w_i}}^\top v_{w_I})
\right],
$$

where:

* $\(v_w\)$ is the input/center embedding of word $\(w\)$
* $\(v'_w\)$ is the output/context embedding of word $\(w\)$
* $\(\sigma(x)\)$ is the sigmoid function
* $\(k\)$ is the number of negative samples
* $\(P_n(w)\)$ is the negative-sampling distribution

The implementation uses PyTorch's `BCEWithLogitsLoss` to optimize the positive and negative scores.

---

### 3. Negative-Sampling Distribution

Negative words are sampled according to a smoothed unigram distribution:

$$
P_n(w)
\propto
f(w)^{0.75},
$$

where $\(f(w)\)$ is the frequency of word $\(w\)$.

The $\(0.75\)$ exponent smooths the raw frequency distribution. Frequent words still have a higher probability of being sampled, but their dominance is reduced compared with sampling directly from the unigram distribution.

The probabilities are normalized so that:

$$
\sum_{w \in V} P_n(w) = 1.
$$

---

## 🚀 Getting Started

### Prerequisites

Python 3.8+ is recommended.

Install the required packages:

```bash
pip install torch nltk scikit-learn matplotlib
```

---

### Clone the Repository

```bash
git clone https://github.com/mahdimeskin/word2vec-skipgram.git
cd word2vec-skipgram
```

---

### Run the Experiment

The complete training and evaluation pipeline is contained in `experiments.ipynb`.

Launch Jupyter:

```bash
jupyter notebook experiments.ipynb
```

The notebook will:

1. Download the NLTK Gutenberg corpus.
2. Load *Alice's Adventures in Wonderland*.
3. Normalize the tokens.
4. Build a vocabulary.
5. Filter low-frequency words.
6. Generate Skip-Gram training pairs.
7. Create a PyTorch `DataLoader`.
8. Construct the negative-sampling distribution.
9. Train the Skip-Gram model.
10. Find nearest neighbors for selected words.
11. Visualize embeddings using t-SNE.

---

## 📚 Dataset

The experiment uses:

**Alice's Adventures in Wonderland** by Lewis Carroll from the NLTK Gutenberg corpus.

The corpus is loaded using:

```python
import nltk
from nltk.corpus import gutenberg

nltk.download("gutenberg")

tokens = [
    token.lower()
    for token in gutenberg.words("carroll-alice.txt")
]
```

Tokens that occur fewer than two times are removed from the vocabulary in the experiment.

---

## 💡 Code Overview

### Vocabulary & Preprocessing (`preprocessing.py`)

The vocabulary builder counts token frequencies and removes words below the specified `min_count` threshold.

```python
from collections import Counter
import torch


def build_vocab(tokens, min_count=1):
    word_counts = Counter(tokens)

    vocab = [
        word
        for word, count in word_counts.items()
        if count >= min_count
    ]

    word_to_idx = {
        word: idx
        for idx, word in enumerate(vocab)
    }

    idx_to_word = {
        idx: word
        for word, idx in word_to_idx.items()
    }

    return word_to_idx, idx_to_word
```

The experiment uses:

```python
word_to_idx, idx_to_word = build_vocab(
    tokens,
    min_count=2
)
```

The negative-sampling distribution is created using the \(0.75\)-powered unigram frequencies:

```python
def create_negative_sampling_distribution(tokens, word_to_idx):
    word_counts = Counter(tokens)

    frequencies = [
        word_counts[word]
        for word, _ in sorted(
            word_to_idx.items(),
            key=lambda x: x[1]
        )
    ]

    weights = (
        torch.tensor(
            frequencies,
            dtype=torch.float32
        ) ** 0.75
    )

    return weights / weights.sum()
```

---

### Skip-Gram Dataset (`dataset.py`)

The context-window generator creates positive center-context pairs.

For example, with:

```text
window_size = 2
```

the model considers up to two words on either side of the center word.

```python
def generate_skipgram_pairs(tokens, window_size=2):
    pairs = []

    for center_idx, center_word in enumerate(tokens):
        start = max(
            0,
            center_idx - window_size
        )

        end = min(
            len(tokens),
            center_idx + window_size + 1
        )

        for context_idx in range(start, end):
            if context_idx != center_idx:
                pairs.append(
                    (
                        center_word,
                        tokens[context_idx]
                    )
                )

    return pairs
```

The generated pairs are converted to vocabulary indices by `SkipGramDataset`.

---

### PyTorch Model (`model.py`)

The model contains two embedding matrices:

```python
self.input_embeddings = nn.Embedding(
    vocab_size,
    embedding_dim
)

self.output_embeddings = nn.Embedding(
    vocab_size,
    embedding_dim
)
```

The input embedding represents the center word, while the output embedding represents context words.

For each positive pair, the model computes a dot product:

$$
s_{pos} =
v_{w_I}^{\top}v'_{w_O}.
$$

For the negative samples, the same operation is performed in a vectorized way using `torch.bmm`.

```python
positive_scores = torch.sum(
    center_vectors * positive_vectors,
    dim=1
)

negative_scores = torch.bmm(
    negative_vectors,
    center_vectors.unsqueeze(2)
).squeeze(2)
```

The resulting tensors have the following shapes:

```text
center_ids       [batch_size]
context_ids      [batch_size]
negative_ids     [batch_size, num_negatives]

center_vectors   [batch_size, embedding_dim]
positive_vectors [batch_size, embedding_dim]
negative_vectors [batch_size, num_negatives, embedding_dim]

positive_scores  [batch_size]
negative_scores  [batch_size, num_negatives]
```

---

## ⚙️ Training Configuration

The current experiment uses:

| Parameter              |                            Value |
| ---------------------- | -------------------------------: |
| Corpus                 | Alice's Adventures in Wonderland |
| Minimum word frequency |                                2 |
| Context window         |                                2 |
| Embedding dimension    |                               50 |
| Batch size             |                              256 |
| Negative samples       |                                5 |
| Epochs                 |                               15 |
| Optimizer              |                             Adam |
| Learning rate          |                            0.002 |

Training automatically uses CUDA when available:

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

---

## 📊 Results & Evaluation

### Word Similarity

After training, the learned input embeddings can be used to find words with similar vector representations.

For example:

```text
Similar to 'alice':
['she', 'thought', 'said', 'little', 'went']

Similar to 'queen':
['king', 'hatter', 'duchess', 'shouted', 'said']
```

Similarity is calculated using cosine similarity between normalized embedding vectors.

```python
import torch.nn.functional as F

embeddings = model.input_embeddings.weight.data

norm_embeddings = F.normalize(
    embeddings,
    p=2,
    dim=1
)
```

These results should be interpreted as **qualitative evidence of structure in the learned embedding space**, rather than as a formal evaluation benchmark.

---

### t-SNE Visualization

The notebook also projects selected 50-dimensional embeddings into two dimensions using t-SNE.

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=10,
    random_state=42
)

vectors_2d = tsne.fit_transform(vectors)
```

The visualization provides a qualitative way to inspect relationships between words in the learned embedding space.

The current experiment visualizes the first 80 words in the vocabulary.

---

## ⚠️ Limitations

This project is an educational implementation of Skip-Gram with Negative Sampling and is **not intended to exactly reproduce every detail of the original Word2Vec implementation**.

Current simplifications include:

* A fixed context window is used rather than dynamic window sampling.
* Frequent-word subsampling is not implemented.
* Negative samples are drawn using PyTorch's `multinomial` sampler.
* Negative samples are not explicitly filtered to prevent them from matching the positive context word.
* The model is trained using Adam rather than the original Word2Vec optimization procedure.
* The experiment uses the relatively small *Alice's Adventures in Wonderland* corpus.
* The current loss uses PyTorch's default mean reduction for the positive and negative binary cross-entropy terms.

These choices keep the implementation relatively simple while preserving the main ideas behind the Skip-Gram with Negative Sampling approach.

---

## 🔬 Possible Future Improvements

Potential extensions include:

* Implement dynamic context-window sampling.
* Implement frequent-word subsampling.
* Improve negative-sample generation to avoid positive targets.
* Implement a learning-rate decay schedule.
* Train on a larger corpus such as WikiText.
* Add quantitative word-vector evaluation benchmarks.
* Compare the learned embeddings with an established Word2Vec implementation.
* Experiment with different embedding dimensions, window sizes, and numbers of negative samples.
* Save and load trained embeddings for downstream NLP tasks.

---

## 📜 References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).
   *Efficient Estimation of Word Representations in Vector Space.*
   [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)

2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013).
   *Distributed Representations of Words and Phrases and their Compositionality.*
   Advances in Neural Information Processing Systems (NeurIPS 2013).

---

## 📄 License

This project is intended for educational and research purposes.
