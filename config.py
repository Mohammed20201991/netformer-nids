# config.py
# Project-wide configuration for NetFormer

import torch

class Config:
    # Paths
    data_path = ""  # User will input at runtime

    # Data preprocessing
    window_size = 100
    stride = 50
    test_split = 0.2
    val_split = 0.2

    # Model architecture
    d_model = 128
    d_ff = 512
    num_heads = 8
    num_layers = 4
    dropout = 0.1

    # Training
    max_epochs = 100
    batch_size = 64
    lr = 1e-4
    weight_decay = 1e-5
    patience = 10

    # Anomaly detection
    threshold_percentile = 95

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate global config
config = Config()