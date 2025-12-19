"""
Utility functions for TGAN (adapted from TimeGAN-pytorch original implementation)

Reference:
Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.
-----------------------------

utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) random_generator: random vector generator
(4) NormMinMax: return data info

Adapted by: [Jordan Baldoceda Perez]
"""

import numpy as np


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """
    Divide train and test data for both original and synthetic data.

    Args:
        data_x: original data (list of sequences)
        data_x_hat: generated data (list of sequences)
        data_t: time lengths for original data
        data_t_hat: time lengths for generated data
        train_rate: ratio of training data (default: 0.8)
    """
    # Divide train/test index for original data
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index for synthetic data
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """
    Returns the sequence lengths and the maximum sequence length in the dataset.

    Args:
        data: list or array of sequences (shape [n_samples, seq_len, features])

    Returns:
        time: list of sequence lengths
        max_seq_len: maximum sequence length
    """
    time = []
    max_seq_len = 0
    for seq in data:
        seq_len = len(seq)
        time.append(seq_len)
        max_seq_len = max(max_seq_len, seq_len)
    return time, max_seq_len


def random_generator(batch_size, z_dim, T_mb, max_seq_len, mean=0.0, std=2.0):
    """
    Generate random latent sequences for the Generator.

    Args:
        batch_size: number of sequences
        z_dim: latent dimension (same as input feature dim)
        T_mb: list of sequence lengths
        max_seq_len: maximum sequence length
        mean: mean of random noise
        std: standard deviation of random noise

    Returns:
        Z_mb: list of numpy arrays of random noise sequences
    """
    Z_mb = []
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.normal(mean, std, [T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp)
    return Z_mb


def NormMinMax(data):
    """
    Apply Min-Max normalization to the dataset.

    Args:
        data: raw data (numpy array of shape [n_samples, seq_len, features])

    Returns:
        norm_data: normalized data
        min_val: per-feature minimums
        max_val: per-feature maximums
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val
