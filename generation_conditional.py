"""
generation_conditional.py
Generate synthetic TimeGAN sequences using SPECIFIC conditions.
This ensures true conditional fidelity (RPM, temperature, current, statics).
"""

import numpy as np
import torch

def generate_from_condition(model, c_time, c_static):
    """
    Generate ONE synthetic sequence using the SAME conditions as a real sequence.
    
    Args:
        model      : trained TimeGAN object
        c_time     : np.ndarray [T, 3]   (Motor_current, Speed(RPM_norm), Temp_norm)
        c_static   : np.ndarray [2]      (weight, distance)
    
    Returns:
        x_syn      : np.ndarray [T, 4]   (synthetic accelerometers)
    """

    device = model.device
    T = c_time.shape[0]

    # -----------------------------
    # 1) Noise Z ~ N(0,1)
    # -----------------------------
    z = np.random.normal(0, 1, size=(1, T, model.opt.z_dim))
    Z = torch.tensor(z, dtype=torch.float32).to(device)

    # -----------------------------
    # 2) Format conditions correctly
    # -----------------------------
    C_time_t = torch.tensor(c_time[np.newaxis, ...], dtype=torch.float32).to(device)

    # static → shape [1,1,2] → then broadcast to [1,T,2]
    c_static_np = np.asarray(c_static, dtype=np.float32).reshape(1, 1, -1)
    C_static_t  = torch.tensor(c_static_np, dtype=torch.float32).to(device)
    C_static_exp = C_static_t.repeat(1, T, 1)
    # -----------------------------
    # 3) Create condition embedding
    # -----------------------------
    with torch.no_grad():

        # Compute condition embedding
        C_embed = model.cond_emb(C_time_t, C_static_exp)   # → [1, T, hidden_dim]

        # Pass Z + embedding to generator
        E_hat = model.netg(Z, C_embed)                     # latent space

        # Supervisor
        H_hat = model.nets(E_hat)

        # Recovery → synthetic accelerometers
        X_hat = model.netr(H_hat)


    return X_hat.cpu().numpy()[0]  # [T,4]


def generate_batch_from_conditions(model, C_time_list, C_static_list, idx_list):
    """
    Generate synthetic sequences for a LIST of real sequences.
    idx_list = indices of real samples to replicate.
    """
    synthetic = []
    for i in idx_list:
        x_syn = generate_from_condition(model, C_time_list[i], C_static_list[i])
        synthetic.append(x_syn.astype(np.float32))
    return synthetic

import numpy as np
import torch

def fast_generate_batch(model, C_time_list, C_static_list, idx_list, batch_size=64):
    """
    Efficient batch generation for conditional TimeGAN.
    Does NOT modify your original .py code.
    """
    device = model.device
    T = C_time_list[0].shape[0]
    synthetic = []

    for start in range(0, len(idx_list), batch_size):
        end = start + batch_size
        batch_idx = idx_list[start:end]

        # 1) Noise batch
        Z = np.random.normal(0, 1, size=(len(batch_idx), T, model.opt.z_dim))
        Z = torch.tensor(Z, dtype=torch.float32).to(device)

        # 2) C_time batch
        C_time_batch = np.array([C_time_list[i] for i in batch_idx])
        C_time_batch = torch.tensor(C_time_batch, dtype=torch.float32).to(device)

        # 3) C_static batch
        C_static_batch = np.array([C_static_list[i] for i in batch_idx])[:, None, :]
        C_static_batch = np.repeat(C_static_batch, T, axis=1)
        C_static_batch = torch.tensor(C_static_batch, dtype=torch.float32).to(device)

        # 4) Forward pass
        with torch.no_grad():
            C_embed = model.cond_emb(C_time_batch, C_static_batch)
            E_hat = model.netg(Z, C_embed)
            H_hat = model.nets(E_hat)
            X_hat = model.netr(H_hat)

        # Append
        for seq in X_hat.cpu().numpy():
            synthetic.append(seq.astype(np.float32))

