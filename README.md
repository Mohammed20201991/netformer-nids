# netformer-nids
Transformer-Based Unsupervised Anomaly Detection for Network Traffic Time-Series

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Research-Active-orange.svg)

## Overview

**NetFormer** is an interpretable Transformer-based framework designed for **Network Intrusion Detection Systems (NIDS)** using **time-series network traffic data**.

The framework leverages the **self-attention mechanism of Transformers** to capture temporal dependencies in network traffic flows while providing **interpretability through attention analysis**.

This project aims to improve anomaly detection performance in modern network environments and provide insights into **which traffic features influence the model's decisions**.

---

## Key Features

* Transformer-based architecture for sequential network traffic analysis
* Attention-based interpretability for explainable intrusion detection
* Designed for **time-series network traffic features**
* Scalable framework for large-scale datasets
* Supports **binary and multi-class intrusion detection**
* Compatible with common network security datasets

---

## Model Architecture

NetFormer utilizes a **Transformer Encoder architecture** consisting of:

1. **Input Embedding Layer**
   Encodes network traffic features into dense vector representations.

2. **Positional Encoding**
   Adds temporal information to preserve sequence order.

3. **Multi-Head Self-Attention**
   Captures dependencies across time steps and network features.

4. **Feed Forward Network (FFN)**
   Non-linear transformation to learn complex feature relationships.

5. **Classification Layer**
   Outputs intrusion detection predictions.

The attention mechanism enables the model to highlight **important features and time steps contributing to anomalies**.

---

## Supported Datasets

The framework can be trained on several widely used network intrusion datasets:

* CICIDS2017
* UNSW-NB15
* NSL-KDD
* Custom network flow datasets

The input data should be formatted as **time-series sequences of network flow features**.

---

## Repository Structure

```
netformer-network-intrusion-detection/
│
├── data/ # Dataset files or preprocessing scripts
│ └── preprocess.py
│
├── models/ # NetFormer model architecture
│ └── netformer.py
│
├── training/ # Training scripts
│ └── train.py
│
├── evaluation/ # Evaluation scripts
│ └── evaluate.py
│
├── utils/ # Utility functions (scalers, encoders, plotting)
│
├── notebooks/ # Experimental notebooks
│
├── requirements.txt # Python dependencies
│
└── README.md # Project documentation
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/username/netformer-network-intrusion-detection.git
cd netformer-network-intrusion-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example dependencies:

```
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

---

## Data Preparation

Place your dataset inside the **data/** directory.

Example structure:

```
data/
│
├── train.csv
├── test.csv
└── preprocess.py
```

Preprocessing may include:

* feature normalization
* sequence construction
* label encoding

---

## Training

Run the training script:

```bash
python training/train.py
```

Configurable parameters:

| Parameter       | Description                                 |
| --------------- | ------------------------------------------- |
| sequence_length | Number of time steps in each input sequence |
| batch_size      | Training batch size                         |
| learning_rate   | Optimizer learning rate                     |
| num_layers      | Number of Transformer layers                |
| num_heads       | Attention heads                             |

---

## Evaluation

To evaluate a trained model:

```bash
python evaluation/evaluate.py
```

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

## Testing New Unseen Data

Test a trained model on new unseen network traffic:

`python evaluation/evaluate.py --unseen data/new_traffic.csv`

## Interpretability

NetFormer provides interpretability by analyzing **attention weights**.

These weights allow researchers and analysts to:

* Identify important network features
* Detect influential time steps
* Understand model decision patterns

Visualization tools can be implemented using:

* matplotlib
* seaborn

---

## Example Workflow

1. Prepare dataset
2. Train NetFormer model
3. Evaluate model performance
4. Visualize attention weights for interpretability

---

## Future Improvements

* Integration with real-time network monitoring systems
* Explainable AI visual dashboards
* Deployment in network security environments
* Support for additional datasets

---

## Citation

If you use this repository in your research, please cite:

```
@article{netformer2026,
  title={NetFormer: An Interpretable Transformer Framework for Network Intrusion Detection in Time-Series Traffic Data},
  author={Mohammed A.S. Al-Hitawi, et al.},
  year={2026}
}
```

---

## License

This project is released under the **MIT License**.

---

## Contact

For questions or collaboration:

Email: [al_hitawe@uofallujah.edu.iq](mailto:al_hitawe@uofallujah.edu.iq)
GitHub: https://github.com/Mohammed20201991
Copyright (c) 2026 Mohammed A.S. Al-Hitawi 
on behalf on authors

