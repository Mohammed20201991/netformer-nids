# Numerical & categorical embedding, positional encoding

import torch
import torch.nn as nn
import numpy as np

class DualStreamEmbedding(nn.Module):
    """Embed numerical and categorical features separately."""
    def __init__(self, num_numerical, cat_cardinalities, d_model):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinalities])
        self.num_proj = nn.Linear(num_numerical, d_model)
        self.d_model = d_model

    def forward(self, x):
        batch, seq_len, _ = x.shape
        num = x[:, :, :self.num_numerical]
        cat = x[:, :, self.num_numerical:].long()
        num_emb = self.num_proj(num)
        cat_emb = sum([emb(cat[:, :, i]) for i, emb in enumerate(self.cat_embeddings)])
        return num_emb + cat_emb

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]