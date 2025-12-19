"""
generation_TGAN.py
Safe mini-batch synthetic data generation for TimeGAN (fixed-length version).
"""

def safe_generation(model, num_samples, batch_size=64, verbose=True):
    """
    Generate synthetic sequences safely using mini-batches.
    Works with fixed-length TimeGAN (Dario Version).
    """

    generated_all = []
    n_batches = (num_samples + batch_size - 1) // batch_size

    if verbose:
        print(f"🧩 Generating {num_samples} samples in {n_batches} batches of {batch_size}...")

    for b in range(n_batches):
        current_n = min(batch_size, num_samples - b * batch_size)

        
        # Generate synthetic mini-batch
        gen_batch = model.generation(num_samples=current_n)


        generated_all.extend(gen_batch)

        if verbose:
            print(f"  ✅ Batch {b+1}/{n_batches} generated ({current_n} samples)")

    return generated_all
