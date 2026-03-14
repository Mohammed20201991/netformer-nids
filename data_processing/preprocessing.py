# Data loading and preprocessing

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """Load CSV dataset. Expects a 'Label' column."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    if 'Label' not in df.columns:
        label_col = [col for col in df.columns if 'label' in col.lower()][0]
        df.rename(columns={label_col: 'Label'}, inplace=True)
    return df

def preprocess(df, config):
    """
    Preprocess the dataframe:
    - Separate numerical/categorical features
    - Normalize numerical features
    - Label encode categorical features
    - Create sliding windows
    """
    print("Preprocessing data...")

    # Separate features and labels
    y = df['Label'].values
    X = df.drop(columns=['Label'])

    # Identify feature types
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    # Encode categoricals
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Normalize numericals
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Combine features
    feature_names = numerical_cols + categorical_cols
    data = X.values.astype(np.float32)

    # Create sliding windows
    windows, labels = [], []
    for i in range(0, len(data) - config.window_size + 1, config.stride):
        win = data[i:i + config.window_size]
        windows.append(win)
        window_labels = y[i:i + config.window_size]
        is_anomaly = 1 if not np.all(window_labels == 'Benign') else 0
        labels.append(is_anomaly)

    windows = np.array(windows)
    labels = np.array(labels)
    print(f"Total windows: {len(windows)}, Anomaly ratio: {labels.mean():.4f}")

    return windows, labels, feature_names, scaler, encoders, numerical_cols, categorical_cols