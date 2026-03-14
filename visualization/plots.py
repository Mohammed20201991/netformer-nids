# Visualization functions for NetFormer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------- Figure 2 ----------
def plot_figure2(window_lengths, f1_scores, fpr_list):
    plt.figure(figsize=(8,5))
    ax1 = plt.gca()
    color1 = 'tab:blue'
    ax1.set_xlabel('Window Length (flows)')
    ax1.set_ylabel('F1-Score', color=color1)
    ax1.plot(window_lengths, f1_scores, color=color1, marker='o', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.75, 0.88)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('False Positive Rate (%)', color=color2)
    ax2.plot(window_lengths, fpr_list, color=color2, marker='s', linestyle='--', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(1.0, 3.0)

    plt.title('Impact of Window Length on NetFormer Performance')
    plt.tight_layout()
    plt.savefig('figure2_window_length.pdf')
    plt.savefig('figure2_window_length.png')
    plt.show()

# ---------- Figure 3 ----------
def plot_figure3(layers_list, f1_scores, times):
    plt.figure(figsize=(8,5))
    ax1 = plt.gca()
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Transformer Layers')
    ax1.set_ylabel('F1-Score', color=color1)
    ax1.bar(layers_list, f1_scores, color=color1, alpha=0.7, width=0.6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.78, 0.87)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Training Time (seconds/epoch)', color=color2)
    ax2.plot(layers_list, times, color=color2, marker='D', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 600)

    plt.title('Impact of Number of Transformer Layers')
    plt.tight_layout()
    plt.savefig('figure3_layers.pdf')
    plt.savefig('figure3_layers.png')
    plt.show()

# ---------- Figure 4 ----------
def plot_figure4(normal_errors, anomalous_errors, threshold):
    plt.figure(figsize=(8,5))
    sns.histplot(normal_errors, bins=50, color='blue', alpha=0.5, label='Normal', stat='density', kde=True)
    sns.histplot(anomalous_errors, bins=50, color='red', alpha=0.5, label='Anomalous', stat='density', kde=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.xlim(0, 0.15)

    inset = inset_axes(plt.gca(), width="40%", height="30%", loc='upper right')
    inset.hist(normal_errors, bins=30, color='blue', alpha=0.5, density=True)
    inset.hist(anomalous_errors, bins=30, color='red', alpha=0.5, density=True)
    inset.axvline(threshold, color='black', linestyle='--', linewidth=1)
    inset.set_xlim(0, 0.1)
    inset.set_ylim(0, 25)
    inset.set_xlabel('Error')
    inset.set_ylabel('Density')
    inset.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    plt.savefig('figure4_error_distribution.pdf')
    plt.savefig('figure4_error_distribution.png')
    plt.show()

# ---------- Figure 5 ----------
def plot_figure5(attention_weights_normal, attention_weights_ddos, attention_weights_slow):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['(a) Normal Sequence', '(b) DDoS Attack', '(c) Slow-Rate DoS Attack']
    attns = [attention_weights_normal, attention_weights_ddos, attention_weights_slow]

    for i, ax in enumerate(axes):
        data = attns[i][0]  # (seq_len, seq_len)
        im = ax.imshow(data, aspect='auto', cmap='viridis')
        ax.set_title(titles[i])
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        if i == 2:
            plt.colorbar(im, ax=ax, label='Attention Weight')

    plt.tight_layout()
    plt.savefig('figure5_attention_maps.pdf')
    plt.savefig('figure5_attention_maps.png')
    plt.show()

# ---------- Figure 6 ----------
def plot_figure6(feature_names, importance_weights, errors):
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importance_weights, xerr=errors, color='steelblue', alpha=0.7, ecolor='black', capsize=3)
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Average Attention Weight')
    plt.title('Feature Importance Based on Attention Weights')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('figure6_feature_importance.pdf')
    plt.savefig('figure6_feature_importance.png')
    plt.show()