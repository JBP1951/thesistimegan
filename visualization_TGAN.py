import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualization(ori_data, generated_data, analysis):
    """Using PCA or t-SNE for generated and original data visualization."""

    # Pick sample size
    anal_sample_no = min([1000, len(ori_data), len(generated_data)])
    idx = np.random.permutation(anal_sample_no)

    # Convert to numpy arrays
    ori_data = np.asarray(ori_data, dtype=object)
    generated_data = np.asarray(generated_data, dtype=object)

    # Subset
    ori_data = [ori_data[i] for i in idx]
    generated_data = [generated_data[i] for i in idx]

    # Ensure same sequence length
    min_len = min(
        min(seq.shape[0] for seq in ori_data),
        min(seq.shape[0] for seq in generated_data)
    )

    # Crop sequences
    ori_data = np.array([seq[:min_len] for seq in ori_data], dtype=np.float32)
    generated_data = np.array([seq[:min_len] for seq in generated_data], dtype=np.float32)

    # Determine size
    no, seq_len, dim = ori_data.shape

    # Flatten for PCA/tSNE
    prep_data = ori_data.reshape(no, -1)
    prep_data_hat = generated_data.reshape(no, -1)

    # ======================================================
    # PCA
    # ======================================================
    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca.fit(prep_data)

        pca_r = pca.transform(prep_data)
        pca_f = pca.transform(prep_data_hat)

        plt.figure(figsize=(6,5))
        plt.scatter(pca_r[:,0], pca_r[:,1], c="red", alpha=0.3, label="Real")
        plt.scatter(pca_f[:,0], pca_f[:,1], c="blue", alpha=0.3, label="Synthetic")
        plt.legend()
        plt.title('PCA: Real vs Synthetic')
        plt.tight_layout()
        plt.show()

    # ======================================================
    # t-SNE
    # ======================================================
    elif analysis == 'tsne':
        print("🌀 Running t-SNE...")

        combined = np.concatenate((prep_data, prep_data_hat), axis=0)
        n_samples = combined.shape[0]
        perp = min(40, max(5, n_samples // 3))

        tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=500)
        tsne_results = tsne.fit_transform(combined)

        plt.figure(figsize=(6,5))
        plt.scatter(tsne_results[:no,0], tsne_results[:no,1], c="red", alpha=0.3, label="Real")
        plt.scatter(tsne_results[no:,0], tsne_results[no:,1], c="blue", alpha=0.3, label="Synthetic")
        plt.legend()
        plt.title('t-SNE: Real vs Synthetic')
        plt.tight_layout()
        plt.show()
