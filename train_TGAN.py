"""
Train_TGAN.py — Adapted for Thesis JORDAN BALDOCEDA

Based on:
"Time-series Generative Adversarial Networks"
Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, NeurIPS 2019.

Adapted by [JORDAN BALDOCEDA] for:
- Rotor dynamics & SHM signals (accelerometers, temperature, encoder)
- Integration with custom surrogate model & experimental data

-----------------------------
Main Workflow:
(1) Load dataset (your signals)
(2) Initialize and train TimeGAN
(3) Optionally visualize or export synthetic signals
"""

import warnings
warnings.filterwarnings("ignore")

import torch

from options_TGAN import Options
from lib.data_preprocess import fit_global_scalers, load_data
from lib.TimeGAN import TimeGAN



def train_TGAN():

    # -------------------------------------------------
    # 1) OPTIONS
    # -------------------------------------------------
    opt = Options().parse()
    opt.conditional = True
    # ===== DRY RUN (TEST 1) =====
    #opt.iteration = 10        # SOLO 10 iteraciones
    #opt.batch_size = 16
    #opt.print_freq = 1


    print("✅ Options loaded")

    # -------------------------------------------------
    # 2) DATA LOADING (GLOBAL NORMALIZATION)
    # -------------------------------------------------
    mat_file = r"C:\Users\Dario\Desktop\ThesiS JBP\Data\all_signals_processed.mat"


    scaler_dyn, scaler_static = fit_global_scalers(
        mat_file,
        accel_names=['Accel1','Accel2','Accel3','Accel4'],
        c_time_names=['Motor_current','Speed','Temperature']
    )

    X, C_time, C_static, feature_names = load_data(
        data_type="mytests",
        seq_len=opt.seq_len,
        file_list=[mat_file],
        scaler_dyn=scaler_dyn,
        scaler_static=scaler_static,
        step=128,
        max_sequences_per_experiment=250
    )




    print("✔ Data loaded correctly")
    print(f"   N sequences: {len(X)}")
    print(f"   X shape:        {X[0].shape}")
    print(f"   C_time shape:   {C_time[0].shape}")
    print(f"   C_static shape: {C_static[0].shape}")

    # -------------------------------------------------
    # 3) INITIALIZE MODEL (CORRECT SIGNATURE)
    # -------------------------------------------------
    model = TimeGAN(
        opt,
        X,
        C_time,
        C_static
    )

    print("✅ TimeGAN initialized")

    # -------------------------------------------------
    # 4) TRAIN
    # -------------------------------------------------
    model.train()
    print("✅ Training completed")


    # ===== GENERATION TEST (TEST 3) =====
    #gen = model.generation(5)
    #print("Generated sample shape:", gen[0].shape)

    return model


if __name__ == "__main__":
    train_TGAN()
