# PyTorch Dataset for network traffic windows

import torch
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    """Dataset for network windows. Optionally uses only normal windows for training."""
    def __init__(self, windows, labels, normal_only=False):
        if normal_only:
            self.windows = windows[labels == 0]
            self.labels = labels[labels == 0]
        else:
            self.windows = windows
            self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)