# Model evaluation utilities

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import torch

def compute_anomaly_scores(model, loader, config):
    """Compute reconstruction error per window."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)
            recon = model(batch)
            mse = torch.mean((recon - batch)**2, dim=(1,2)).cpu().numpy()
            errors.extend(mse)
    return np.array(errors)

def find_threshold(val_errors, val_labels, percentile=95):
    """Set threshold at percentile of normal validation errors."""
    normal_errors = val_errors[val_labels == 0]
    return np.percentile(normal_errors, percentile)

def evaluate_model(test_errors, test_labels, threshold):
    """Compute evaluation metrics based on threshold."""
    preds = (test_errors > threshold).astype(int)
    precision = precision_score(test_labels, preds, zero_division=0)
    recall = recall_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)
    auc_roc = roc_auc_score(test_labels, test_errors)
    auc_pr = average_precision_score(test_labels, test_errors)
    return precision, recall, f1, auc_roc, auc_pr, preds