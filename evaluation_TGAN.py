import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    r2_score
)
from scipy.stats import entropy, pearsonr, ks_2samp


def flatten_sequences(real_seqs, synth_seqs, max_samples=None, max_points=50000):
    n = min(len(real_seqs), len(synth_seqs))
    if max_samples is not None:
        n = min(n, max_samples)

    real_flat  = np.concatenate(real_seqs[:n], axis=0)
    synth_flat = np.concatenate(synth_seqs[:n], axis=0)

    min_len = min(len(real_flat), len(synth_flat))
    real_flat, synth_flat = real_flat[:min_len], synth_flat[:min_len]

    # solo para estabilidad estadística
    if len(real_flat) > max_points:
        idx = np.random.choice(len(real_flat), max_points, replace=False)
        real_flat  = real_flat[idx]
        synth_flat = synth_flat[idx]

    return real_flat, synth_flat



def evaluate_timegan(real_seqs, synth_seqs, max_samples=2000, verbose=True):
    """
    Improved evaluation for TimeGAN real vs synthetic.
    Adds KS test and improved KL-divergence.
    """

    # Flatten
    real_flat, synth_flat = flatten_sequences(real_seqs, synth_seqs, max_samples)
    n_features = real_flat.shape[1]

    # -------------------------------
    # Error Metrics (base version)
    # -------------------------------
    mse = mean_squared_error(real_flat, synth_flat)
    mae = mean_absolute_error(real_flat, synth_flat)
    mape = mean_absolute_percentage_error(real_flat + 1e-8,
                                          synth_flat + 1e-8)

    if np.any(real_flat <= -1) or np.any(synth_flat <= -1):
        msle = np.nan
    else:
        msle = mean_squared_log_error(real_flat + 1e-8,
                                      np.abs(synth_flat) + 1e-8)

    r2 = r2_score(real_flat, synth_flat)

    # -------------------------------
    # Pearson Correlation per feature
    # -------------------------------
    per_feature_corr = []
    for f in range(n_features):
        corr, _ = pearsonr(real_flat[:, f], synth_flat[:, f])
        per_feature_corr.append(corr)

    mean_corr = np.nanmean(per_feature_corr)

    # -------------------------------
    # KL Divergence (per feature)
    # -------------------------------
    kl_values = []

    for f in range(n_features):
        p_hist, _ = np.histogram(real_flat[:, f], bins=50, density=True)
        q_hist, _ = np.histogram(synth_flat[:, f], bins=50, density=True)
        kl = entropy(p_hist + 1e-8, q_hist + 1e-8)
        kl_values.append(kl)

    mean_kl = np.mean(kl_values)

    # -------------------------------
    # KS Test (MOST IMPORTANT)
    # -------------------------------
    ks_p_values = []

    for f in range(n_features):
        stat, p = ks_2samp(real_flat[:, f], synth_flat[:, f])
        ks_p_values.append(p)

    ks_mean = np.mean(ks_p_values)

    # -------------------------------
    # PACK RESULTS
    # -------------------------------
    results = {
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape,
        "MSLE": msle,
        "R2": r2,
        "Mean_Corr": mean_corr,
        "KL_Mean": mean_kl,
        "KS_Mean": ks_mean
    }

   

    if verbose:
        print("\n⚠️ NOTE:")
        print("Pointwise error metrics (MSE, MAE, R²) are reported for completeness only.")
        print("They are NOT fully representative of generative fidelity in GAN-based time-series models.")
        print("Distributional metrics (KS, KL, correlation) should be prioritized.\n")
        print("\n📊 [TimeGAN Evaluation Metrics]")
        print("-----------------------------------")
        for k, v in results.items():
            print(f"{k:>12s}: {v:.6f}")

        print("\n🔍 KS Test per Feature (p > 0.05 means similar distributions):")
        for f, p in enumerate(ks_p_values):
            warn = "⚠️" if p < 0.05 else ""
            print(f"  Feature {f}: p = {p:.4f} {warn}")

        print("-----------------------------------\n")

    return results
