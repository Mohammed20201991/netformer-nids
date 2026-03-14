# Run NetFormer experiment and test on new unseen data

from config import Config
from data.preprocess import load_data, preprocess, TrafficDataset
from models.netformer import NetFormer
from training.trainer import train_model
from evaluation.evaluate import compute_anomaly_scores, find_threshold, evaluate_model
from baselines.baselines import run_baselines
from visualization import plots
from torch.utils.data import DataLoader
import torch
import numpy as np

def run_experiment(config):
    # 1. Load & preprocess
    df = load_data(config.data_path)
    windows, labels, feature_names, scaler, encoders, num_cols, cat_cols = preprocess(df, config)

    # Split data temporally
    n = len(windows)
    train_end = int(n * (1 - config.test_split - config.val_split))
    val_end = train_end + int(n * config.val_split)
    train_windows, train_labels = windows[:train_end], labels[:train_end]
    val_windows, val_labels = windows[train_end:val_end], labels[train_end:val_end]
    test_windows, test_labels = windows[val_end:], labels[val_end:]

    train_dataset = TrafficDataset(train_windows, train_labels, normal_only=True)
    val_dataset = TrafficDataset(val_windows, val_labels)
    test_dataset = TrafficDataset(test_windows, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    cat_cardinalities = [len(encoders[col].classes_) for col in cat_cols]
    model = NetFormer(config, len(num_cols), cat_cardinalities).to(config.device)

    # 2. Train model
    print("Training NetFormer...")
    model, history = train_model(model, train_loader, val_loader, config)

    # 3. Compute reconstruction errors
    val_errors = compute_anomaly_scores(model, val_loader, config)
    test_errors = compute_anomaly_scores(model, test_loader, config)

    # 4. Determine threshold
    threshold = find_threshold(val_errors, val_labels, percentile=config.threshold_percentile)
    print(f"Threshold = {threshold:.4f}")

    # 5. Evaluate on test set
    precision, recall, f1, auc_roc, auc_pr, preds = evaluate_model(test_errors, test_labels, threshold)
    print(f"Test Results: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")

    # 6. Run baselines
    baseline_results = run_baselines(train_windows[train_labels==0], train_labels[train_labels==0],
                                     test_windows, test_labels, config)
    print("Baseline AUC-ROC:", baseline_results)

    return model, scaler, encoders, threshold, num_cols, cat_cols

def test_unseen(model, scaler, encoders, threshold, num_cols, cat_cols, new_data_path, config):
    """Test NetFormer on new unseen CSV dataset."""
    df_new = load_data(new_data_path)
    windows, labels, _, _, _, _, _ = preprocess(df_new, config)
    dataset = TrafficDataset(windows, labels)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    errors = compute_anomaly_scores(model, loader, config)
    preds = (errors > threshold).astype(int)
    anomaly_ratio = preds.mean()
    print(f"Unseen Data: {len(windows)} windows, {anomaly_ratio*100:.2f}% predicted anomalies")
    return preds, errors

if __name__ == "__main__":
    config = Config()
    config.data_path = input("Enter training dataset CSV path: ").strip()

    model, scaler, encoders, threshold, num_cols, cat_cols = run_experiment(config)

    # Test on new unseen data
    new_path = input("Enter new unseen dataset CSV path to test: ").strip()
    preds, errors = test_unseen(model, scaler, encoders, threshold, num_cols, cat_cols, new_path, config)