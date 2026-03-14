# NetFormer Autoencoder Model

import torch.nn as nn
from .embedding import DualStreamEmbedding, PositionalEncoding
from .transformer import TransformerEncoder
from .decoder import Decoder

class NetFormer(nn.Module):
    """Embedding + PositionalEncoding + Encoder + Decoder."""
    def __init__(self, config, num_numerical, cat_cardinalities):
        super().__init__()
        self.embedding = DualStreamEmbedding(num_numerical, cat_cardinalities, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, max_len=config.window_size)
        self.encoder = TransformerEncoder(config.num_layers, config.d_model, config.num_heads, config.d_ff, config.dropout)
        self.decoder = Decoder(config.num_layers, config.d_model, config.num_heads, config.d_ff, config.dropout,
                               output_dim=num_numerical + len(cat_cardinalities))

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.pos_enc(emb)
        latent = self.encoder(emb)
        recon = self.decoder(latent)
        return recon

    def get_attention_weights(self, x):
        """Return attention weights from first encoder layer (for visualization)."""
        emb = self.embedding(x)
        emb = self.pos_enc(emb)
        attn_layer = self.encoder.layers[0].self_attn
        _, attn_weights = attn_layer(emb, emb, emb)
        return attn_weights