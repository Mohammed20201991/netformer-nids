# Train and evaluate simple baseline models for comparison

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

def run_baselines(X_train, y_train, X_test, y_test, config):
    results = {}

    # Isolation Forest
    print("Training Isolation Forest...")
    if_model = IsolationForest(contamination=0.1, random_state=42)
    if_model.fit(X_train.reshape(X_train.shape[0], -1))
    if_score = -if_model.decision_function(X_test.reshape(X_test.shape[0], -1))
    results['IF'] = {'AUC-ROC': roc_auc_score(y_test, if_score)}

    # One-Class SVM
    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf')
    ocsvm.fit(X_train.reshape(X_train.shape[0], -1))
    ocsvm_score = -ocsvm.decision_function(X_test.reshape(X_test.shape[0], -1))
    results['OCSVM'] = {'AUC-ROC': roc_auc_score(y_test, ocsvm_score)}

    # LSTM Autoencoder (simple)
    print("Training LSTM Autoencoder...")
    class LSTMAE(nn.Module):
        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        def forward(self, x):
            out, (h, c) = self.encoder(x)
            out, _ = self.decoder(out)
            return out

    lstm_model = LSTMAE(X_train.shape[2]).to(config.device)
    optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    lstm_model.train()
    for epoch in range(10):
        for i in range(0, len(X_train), config.batch_size):
            batch = torch.tensor(X_train[i:i+config.batch_size], dtype=torch.float32).to(config.device)
            recon = lstm_model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.device)
        recon_test = lstm_model(test_tensor)
        lstm_errors = torch.mean((recon_test - test_tensor)**2, dim=(1,2)).cpu().numpy()
    results['LSTM-AE'] = {'AUC-ROC': roc_auc_score(y_test, lstm_errors)}

    return results