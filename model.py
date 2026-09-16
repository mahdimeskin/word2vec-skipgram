import torch
import torch.nn as nn

class Word2VecSkipGram(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        self.input_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        
        self.output_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        
    def forward(
        self,
        center_ids,
        context_ids,
        negative_ids
    ):
        # center_ids: [batch_size]
        # context_ids: [batch_size]
        # negative_ids: [batch_size, num_negatives]
        
        center_vectors = self.input_embeddings(center_ids)
        
        positive_vectors = self.output_embeddings(context_ids)
        
        negative_vectors = self.output_embeddings(negative_ids)
        
        positive_scores = torch.sum(
            center_vectors * positive_vectors,
            dim=1
        )
        
        negative_scores = torch.bmm(
            negative_vectors,
            center_vectors.unsqueeze(2)
        ).squeeze(2)
        
        return positive_scores, negative_scores
    
    def loss(
        self,
        positive_scores,
        negative_scores
    ):
        positive_labels = torch.ones_like(
            positive_scores
        )

        negative_labels = torch.zeros_like(
            negative_scores
        )

        loss_fn = nn.BCEWithLogitsLoss()

        positive_loss = loss_fn(
            positive_scores,
            positive_labels
        )

        negative_loss = loss_fn(
            negative_scores,
            negative_labels
        )

        return positive_loss + negative_loss

