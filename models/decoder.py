# Transformer decoder for reconstruction

import torch.nn as nn
from .transformer import TransformerEncoderLayer

class Decoder(nn.Module):
    """Simple decoder mirrors encoder to reconstruct input."""
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)