"""
timegan_TGAN.py
Adapted (minimal changes) TimeGAN runner based on the PyTorch reimplementation.

Keep structure and method names identical to original; minimal adaptions:
 - imports adjusted to local utils/data/module names used in this thesis
 - robust device detection
 - safe renormalization in `generation()`

 Note: Use original data as training set to generater synthetic data (time-series)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Use your data_preprocess batch generator (must provide batch_generator)
from lib.data_preprocess import batch_generator, batch_generator_conditional

# Use the utils file adapted for your repo (utils_TGAN.py)
from utils_TGAN import extract_time, random_generator, NormMinMax

# Import the adapted model (model_TGAN.py)
from lib.model_TGAN import Encoder, Recovery, Generator, Discriminator, Supervisor,CondEmbedding




class BaseModel():
  """ Base Model for TimeGAN (kept same structure as original) """
  def __init__(self, opt, ori_data):
    # Seed for deterministic behavior
    self.seed(getattr(opt, "manualseed", -1))

    # Initialize variables
    self.opt = opt

    # Tus datos YA vienen normalizados desde data_preprocess.py
    # No volvemos a normalizar aquí (evita MemoryError y doble escalado)
        # Tus datos YA vienen normalizados
    self.min_val = None
    self.max_val = None

    # ------------------------------
    # ¿Es dict (condicional) o array?
    # ------------------------------
    if isinstance(ori_data, dict):
        # caso condicional
        self.conditional   = True
        
        self.X_all         = ori_data["X"]         # lista de [L, 4]
        self.C_time_all    = ori_data["C_time"]    # lista de [L, 3]
        self.C_static_all  = ori_data["C_static"]  # lista de [2]
    else:
        # caso antiguo (no condicional)
        self.conditional   = False
        
        self.X_all         = ori_data              # lista/array de [L, dim]
        self.C_time_all    = None
        self.C_static_all  = None

    self.opt.conditional = self.conditional
    #self.opt.conditional = True          # ← ESTA ES LA LÍNEA CRÍTICA

    # ------------------------------
    # Dimensión de datos a generar (X)
    # ------------------------------
    first_seq = np.asarray(self.X_all[0])
    self.x_dim = first_seq.shape[1]

    print(f"[INFO] Detected data feature dim (x_dim) = {self.x_dim}")
    # Muy importante: dejamos z_dim (noise) como en Options
    print(f"[INFO] Using latent noise dim (z_dim)   = {self.opt.z_dim}")

    # ------------------------------
    # Dimensiones de condicionales
    # ------------------------------
    if self.conditional:
        self.c_time_dim   = self.C_time_all[0].shape[1]   # 3 → [Motor, Speed, Temp]
        self.c_static_dim = self.C_static_all[0].shape[-1]# 2 → [weight, distance]
    else:
        self.c_time_dim   = 0
        self.c_static_dim = 0

    # Guardamos en opt para que las redes puedan usarlos
    self.opt.x_dim        = self.x_dim
    self.opt.c_time_dim   = self.c_time_dim
    self.opt.c_static_dim = self.c_static_dim
    self.opt.conditional  = self.conditional

    # Tiempos (solo dependen de X)
    self.ori_time, self.max_seq_len = extract_time(self.X_all)
    self.data_num = len(self.X_all)



    # Train/test directories for checkpoints/logs
    self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
    self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

    # Device (use GPU if requested and available)
    if getattr(self.opt, "device", "gpu") != 'cpu' and torch.cuda.is_available():
      self.device = torch.device("cuda:0")
    else:
      self.device = torch.device("cpu")

    # ADDED TO SOLVE DEBUGGING PART OF LOSS
    # losses (same as original)
    self.l_mse = nn.MSELoss()
    self.l_r = nn.L1Loss()
    self.l_bce = nn.BCELoss()


  

  def seed(self, seed_value):
    """Seed RNGs for reproducibility"""
    if seed_value == -1:
      return
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True

  def save_weights(self, epoch):
    """Save nets' weights for current epoch"""
    weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
    if not os.path.exists(weight_dir):
      os.makedirs(weight_dir)

    torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
               '%s/netE.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()},
               '%s/netR.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
               '%s/netG.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
               '%s/netD.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()},
               '%s/netS.pth' % (weight_dir))

  def sample_batch(self):

    
    # ---------------------------------------------------------
    #   CONDITIONAL TIMEGAN
    # ---------------------------------------------------------
    if self.opt.conditional:
      X_mb, C_time_mb, C_static_mb, T_mb = batch_generator_conditional(
          self.X_all,
          self.C_time_all,
          self.C_static_all,
          self.ori_time,
          self.opt.batch_size
      )

      # Guardamos batch para debug
      self.X0 = X_mb
      self.T  = T_mb

      B = X_mb.shape[0]
      T = X_mb.shape[1]

      # ---------------------------------------------------------
      # 1) FORMA DE C_time_mb  → debe ser [B, T, c_time_dim]
      # ---------------------------------------------------------
      C_time_mb = np.asarray(C_time_mb)

      # Garantizar forma [B, T, c_time_dim]
      if C_time_mb.ndim == 1:
          # caso extremo: batch_size=1 y NumPy colapsó → [c_time_dim]
          C_time_mb = C_time_mb.reshape(1, T, -1)

      elif C_time_mb.ndim == 2:
          # caso raro: [B, c_time_dim] → expandir en tiempo
          C_time_mb = C_time_mb[:, None, :].repeat(T, axis=1)

      elif C_time_mb.ndim == 3:
          # caso correcto → [B, T, c_time_dim]
          pass

      else:
          raise ValueError(f"C_time_mb shape inválida: {C_time_mb.shape}")


      # ---------------------------------------------------------
      # 2) FORMA DE C_static_mb  → debe ser [B, 1, c_static_dim]
      # ---------------------------------------------------------
      C_static_mb = np.asarray(C_static_mb)

      # Garantizar forma [B, 1, c_static_dim]
      if C_static_mb.ndim == 1:
          # caso batch_size = 1 → [2] → [1,1,2]
          C_static_mb = C_static_mb.reshape(1, 1, -1)

      elif C_static_mb.ndim == 2:
          # caso normal → [B,2] → [B,1,2]
          C_static_mb = C_static_mb[:, None, :]

      elif C_static_mb.ndim == 3:
          # ya correcto → [B,1,2]
          pass

      else:
          raise ValueError(f"C_static_mb shape inválida: {C_static_mb.shape}")


      # ---------------------------------------------------------
      # Convertimos a tensores (AHORA sí sin problemas)
      # ---------------------------------------------------------
      self.X            = torch.tensor(X_mb,        dtype=torch.float32).to(self.device)
      self.C_time_mb    = torch.tensor(C_time_mb,   dtype=torch.float32).to(self.device)
      self.C_static_mb  = torch.tensor(C_static_mb, dtype=torch.float32).to(self.device)

      # ===== DEBUG SHAPES (TEST 2) =====
      #if np.random.rand() < 0.01:   # imprime solo a veces
       #   print("\n[DEBUG SHAPES]")
        #  print("X:", self.X.shape)
         # print("C_time:", self.C_time_mb.shape)
          #print("C_static:", self.C_static_mb.shape)


    #return

    else:

    # ---------------------------------------------------------
    #   NON-CONDITIONAL TIMEGAN
    # ---------------------------------------------------------
      X_mb, T_mb = batch_generator(
          self.X_all,
          self.ori_time,
          self.opt.batch_size
      )

      self.X0 = X_mb
      self.T  = T_mb

      self.X            = torch.tensor(X_mb, dtype=torch.float32).to(self.device)
      self.C_time_mb    = None
      self.C_static_mb  = None







  def train_one_iter_er(self):
    """ Train encoder & recovery for one mini-batch """
    self.nete.train()
    self.netr.train()

    self.sample_batch()   # ← aquí obtenemos X, C_time, C_static, T

    loss = self.optimize_params_er()
    return loss


  def train_one_iter_er_(self):
    """ Train encoder & recovery (alternate objective) for one mini-batch """
    self.nete.train()
    self.netr.train()

    self.sample_batch()

    self.optimize_params_er_()
    


  def train_one_iter_s(self):
    """ Train supervisor """
    self.nets.train()

    self.sample_batch()

    s_loss = self.optimize_params_s()
    return s_loss



  def train_one_iter_g(self):
    """ Train generator-related parts """
    self.netg.train()

    self.sample_batch()

    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)
    self.Z = np.asarray(self.Z)  # make sure numpy array
    g_loss  = self.optimize_params_g()
    return g_loss


  def train_one_iter_d(self):
    # 🔥 CLEAN GPU MEMORY EACH CRITIC STEP
    torch.cuda.empty_cache()
    """ Train discriminator """
    self.netd.train()

    self.sample_batch()

    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)
    self.Z = np.asarray(self.Z)
    d_loss  = self.optimize_params_d()
    return d_loss



  def train(self):
   
    print("=== PRETRAINING: Embedding + Recovery ===")
    for it in range(self.opt.iteration):
        er_loss = self.train_one_iter_er()

        if it % self.opt.print_freq == 0 or it == self.opt.iteration - 1:
            print(f"[ER] Iter {it}/{self.opt.iteration} | ER Loss = {er_loss:.6f}")

    print("=== PRETRAINING: Supervisor ===")
    for it in range(self.opt.iteration):
        s_loss = self.train_one_iter_s()

        if it % self.opt.print_freq == 0 or it == self.opt.iteration - 1:
            print(f"[S] Iter {it}/{self.opt.iteration} | Supervisor Loss = {s_loss:.6f}")

    print("=== ADVERSARIAL TRAINING (WGAN-GP) ===")
    for it in range(self.opt.iteration):

        # ---- Train Critic (D) ----
        for _ in range(self.opt.n_critic):
            d_loss = self.train_one_iter_d()

        # ---- Train Generator ----
        g_loss = self.train_one_iter_g()

        if it % self.opt.print_freq == 0 or it == self.opt.iteration - 1:
            print("----------------------------------------------")
            print(f"[WGAN] Iter {it}/{self.opt.iteration}")
            print(f"  D Loss = {d_loss:.6f}")
            print(f"  G Loss = {g_loss:.6f}")
            print("----------------------------------------------")

    self.save_weights(self.opt.iteration)
    self.generated_data = self.generation(self.opt.batch_size)
    print("Finish Synthetic Data Generation")



  def generation(self, num_samples, mean=0.0, std=1.0):
      import numpy as np 
      
      if num_samples == 0:
          return None

      T = self.max_seq_len
      T_mb = [T] * num_samples

      # 1) Ruido Z
      Z = random_generator(
          num_samples,
          self.opt.z_dim,
          T_mb,
          self.max_seq_len,
          mean,
          std
      )
      Z = np.asarray(Z, dtype=np.float32)  # evita el warning molesto

      Z = torch.tensor(Z, dtype=torch.float32).to(self.device)

      # ---------------------------------------------------
      # 2) MODO CONDICIONAL: samplear condiciones reales
      # ---------------------------------------------------
      if getattr(self.opt, "conditional", False):

         
          # Elegimos índices aleatorios de la base real
          idx = np.random.permutation(self.data_num)[:num_samples]

          # C_time: [B, T, 3]  (recortamos/ajustamos a T=max_seq_len)
          C_time_mb = np.asarray(
              [self.C_time_all[i][:T] for i in idx],
              dtype=np.float32
          )  # [B, T, 3]

          # C_static: [B, 2] -> [B, 1, 2] -> [B, T, 2]
          C_static_mb = np.asarray(
              [self.C_static_all[i] for i in idx],
              dtype=np.float32
          )  # [B, 2]

          if C_static_mb.ndim == 2:
              C_static_mb = C_static_mb[:, np.newaxis, :]   # [B,1,2]

          C_static_exp = np.repeat(C_static_mb, T, axis=1)  # [B,T,2]

          # Pasar a torch
          C_time_mb = torch.tensor(C_time_mb, dtype=torch.float32).to(self.device)
          C_static_exp = torch.tensor(C_static_exp, dtype=torch.float32).to(self.device)

          # 1) Crear embedding de condiciones
          C_embed = self.cond_emb(C_time_mb, C_static_exp)

          # 2) Pasar Z + C_embed al generador
          E_hat = self.netg(Z, C_embed)


      else:
          # -----------------------------------------------
          # 2-bis) MODO NO CONDICIONAL
          # -----------------------------------------------
          E_hat = self.netg(Z)

      # 3) Supervisor + Recovery
      H_hat = self.nets(E_hat)
      X_hat = self.netr(H_hat).cpu().detach().numpy()  # [B, T, x_dim]

      generated_data = [X_hat[i] for i in range(num_samples)]
      return generated_data





class TimeGAN(BaseModel):

    def __init__(self, opt, X_list, C_time_list=None, C_static_list=None):

        # PREPARAR DICCIONARIO PARA BASEMODEL
      if getattr(opt, "conditional", False):
            data_dict = {
                "X": X_list,
                "C_time": C_time_list,
                "C_static": C_static_list
            }
      else:
            data_dict = X_list

        # BASEMODEL RECIBE TODO
      super(TimeGAN, self).__init__(opt, data_dict)
      # Ahora que BaseModel ya extrajo las dims reales, recalculamos input_dim
      if self.conditional:
          # Encoder recibe [X, C_embed] → x_dim + hidden_dim
          self.opt.input_dim = self.opt.x_dim + self.opt.hidden_dim
      else:
          # Encoder recibe solo X → x_dim
          self.opt.input_dim = self.opt.x_dim

      # 🔥 enable instance noise
     

        # Crear redes como siempre
      self.epoch = 0
      self.total_steps = 0
      self.nete = Encoder(opt).to(self.device)
      self.netr = Recovery(opt).to(self.device)
      self.netg = Generator(opt).to(self.device)
      self.netd = Discriminator(opt).to(self.device)
      self.nets = Supervisor(opt).to(self.device)

      self.instance_noise = True
      self.netd.instance_noise = True


      # 🔥 NEW CONDITION EMBEDDING NETWORK
      self.cond_emb = CondEmbedding(
          opt.c_time_dim,
          opt.c_static_dim,
          opt.hidden_dim
      ).to(self.device)


      # Optionally load pre-trained models (kept same)
      if getattr(self.opt, "resume", '') != '':
        print("\nLoading pre-trained networks.")
        self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
        self.nete.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netE.pth'))['state_dict'])
        self.netr.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netR.pth'))['state_dict'])
        self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
        self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
        self.nets.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netS.pth'))['state_dict'])
        print("\tDone.\n")

      # losses (same as original)
      self.l_mse = nn.MSELoss()
      self.l_r = nn.L1Loss()
      self.l_bce = nn.BCELoss()

      # Setup optimizer (requires opt.lr and opt.beta1 — keep them in options)
      if getattr(self.opt, "isTrain", True):
        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()

        # allow fallback defaults if not present in opt
        lr = getattr(self.opt, "lr", 0.001)
        beta1 = getattr(self.opt, "beta1", 0.9)

                # Encoder + Recovery optimizers
        self.optimizer_e = optim.Adam(self.nete.parameters(),
                                      lr=getattr(self.opt, "lr_e", lr),
                                      betas=(beta1, 0.9))
        self.optimizer_r = optim.Adam(self.netr.parameters(),
                                      lr=getattr(self.opt, "lr_r", lr),
                                      betas=(beta1, 0.9))

        # Supervisor optimizer
        self.optimizer_s = optim.Adam(self.nets.parameters(),
                                      lr=getattr(self.opt, "lr_s", lr),
                                      betas=(beta1, 0.9))

        # Generator optimizer (separate LR)
        self.optimizer_g = optim.Adam(self.netg.parameters(),
                                      lr=getattr(self.opt, "lr_g", lr),
                                      betas=(beta1, 0.9))

        # Discriminator optimizer (separate LR)
        self.optimizer_d = optim.Adam(self.netd.parameters(),
                                      lr=getattr(self.opt, "lr_d", lr),
                                      betas=(beta1, 0.9))
        
        # -----------------------------
        # Add LR schedulers (CTGAN-MSIN paper)
        # -----------------------------
        from torch.optim.lr_scheduler import StepLR

        DECAY_STEPS = 10000  # recomendado

        self.scheduler_e = StepLR(self.optimizer_e, step_size=DECAY_STEPS, gamma=0.96)
        self.scheduler_r = StepLR(self.optimizer_r, step_size=DECAY_STEPS, gamma=0.96)
        self.scheduler_s = StepLR(self.optimizer_s, step_size=DECAY_STEPS, gamma=0.96)
        self.scheduler_g = StepLR(self.optimizer_g, step_size=DECAY_STEPS, gamma=0.96)
        self.scheduler_d = StepLR(self.optimizer_d, step_size=DECAY_STEPS, gamma=0.96)



    # -----------------------
    # Forward helpers (same names as original)
    # -----------------------
    
    def forward_e(self):

      if self.conditional:

          # 🔥 1) Obtener embedding no lineal de condiciones
          C_embed = self.cond_emb(self.C_time_mb, self.C_static_mb)  
          # C_embed → [B, T, hidden_dim]

          # 🔥 2) Concatenar X solo con el embed, no con condiciones crudas
          X_full = torch.cat([self.X, C_embed], dim=2)

      else:
          X_full = self.X

      # 🔥 3) Encoder procesa X + EMBEDDING
      self.H = self.nete(X_full)



    def forward_g(self):

      self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)

      if self.conditional:

          # 🔥 Embedding de condiciones
          C_embed = self.cond_emb(self.C_time_mb, self.C_static_mb)

          # 🔥 El generador recibe ruido + embedding
          self.E_hat = self.netg(self.Z, C_embed)

      else:
          self.E_hat = self.netg(self.Z)






    def forward_dg(self):

      if self.conditional:
          C_embed = self.cond_emb(self.C_time_mb, self.C_static_mb)

          self.Y_fake   = self.netd(self.H_hat, C_embed)
          self.Y_fake_e = self.netd(self.E_hat, C_embed)

      else:
          self.Y_fake   = self.netd(self.H_hat)
          self.Y_fake_e = self.netd(self.E_hat)



    def forward_rg(self):
      self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
      self.H_supervise = self.nets(self.H)

    def forward_sg(self):
      self.H_hat = self.nets(self.E_hat)

    def forward_d(self):

      if self.conditional:

          # 🔥 Embedding para pasar al discriminador
          C_embed = self.cond_emb(self.C_time_mb, self.C_static_mb)

          self.Y_real   = self.netd(self.H,     C_embed)
          self.Y_fake   = self.netd(self.H_hat, C_embed)
          self.Y_fake_e = self.netd(self.E_hat, C_embed)

      else:
          self.Y_real   = self.netd(self.H)
          self.Y_fake   = self.netd(self.H_hat)
          self.Y_fake_e = self.netd(self.E_hat)



    # -----------------------
    # Backward helpers (same as original)
    # -----------------------
    def forward_er(self):
      if self.conditional:
        #B, T, _ = self.X.shape
        #C_static_rep = self.C_static_mb.repeat(1, T, 1)
        C_embed = self.cond_emb(self.C_time_mb, self.C_static_mb)
        X_full = torch.cat([self.X, C_embed], dim=2)
      else:
        X_full = self.X

      self.H = self.nete(X_full)
      self.X_tilde = self.netr(self.H)


    def backward_er_(self):
      self.err_er_ = self.l_mse(self.X_tilde, self.X)
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
      self.err_er.backward(retain_graph=True)

    def backward_g(self):
      """
      Generator loss with:
      - WGAN adversarial loss
      - Mean/Std matching (V1, V2)
      - NEW: Feature Matching loss (FM)
      - Supervisor consistency
      """

      # ----------------------------------------
      # 1) Adversarial losses
      # ----------------------------------------
      adv_fake = - self.Y_fake.mean()
      adv_fake_e = - self.Y_fake_e.mean()
      self.err_g_adv = adv_fake + self.opt.w_gamma * adv_fake_e

      # ----------------------------------------
      # 2) V1 / V2 (mean and std matching)
      # ----------------------------------------
      real_std = torch.std(self.X, dim=[0,1])
      fake_std = torch.std(self.X_hat, dim=[0,1])
      self.err_g_V1 = torch.mean(torch.abs(real_std - fake_std))

      real_mean = torch.mean(self.X, dim=[0,1])
      fake_mean = torch.mean(self.X_hat, dim=[0,1])
      self.err_g_V2 = torch.mean(torch.abs(real_mean - fake_mean))

      # ----------------------------------------
      # 3) NEW: Feature Matching Loss
      #    Compare latent real H vs generated H_hat
      # ----------------------------------------
      H_real_mean = self.H.mean(dim=[0,1])
      H_fake_mean = self.H_hat.mean(dim=[0,1])
      self.err_g_FM = torch.mean(torch.abs(H_real_mean - H_fake_mean))

      # ----------------------------------------
      # 4) Supervisor loss
      # ----------------------------------------
      self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])

      # ----------------------------------------
      # 5) Total Generator Loss
      # ----------------------------------------
      self.err_g = (
            self.err_g_adv
          + self.err_g_V1 * self.opt.w_g
          + self.err_g_V2 * self.opt.w_g
          + self.err_g_FM * self.opt.w_fm      # NEW FEATURE MATCHING WEIGHT
          + torch.sqrt(self.err_s)
      )

      # Backprop
      self.err_g.backward(retain_graph=True)

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(self.netg.parameters(), max_norm=1.0)


      #print("Loss G (total): ", self.err_g)

      '''
      self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))
      self.err_g_U_e = self.l_bce(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
      self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(self.X_hat,[0])[1] + 1e-6) - torch.sqrt(torch.std(self.X,[0])[1] + 1e-6)))
      self.err_g_V2 = torch.mean(torch.abs((torch.mean(self.X_hat,[0])[0]) - (torch.mean(self.X,[0])[0])))
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_g = self.err_g_U + \
                   self.err_g_U_e * self.opt.w_gamma + \
                   self.err_g_V1 * self.opt.w_g + \
                   self.err_g_V2 * self.opt.w_g + \
                   torch.sqrt(self.err_s)
      self.err_g.backward(retain_graph=True)
      print("Loss G: ", self.err_g)'''

    def backward_s(self):
      self.err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:])
      self.err_s.backward(retain_graph=True)
      #print("Loss S: ", self.err_s)

    # GRADIENT PENALTY NEW
    def gradient_penalty(self, real, fake):

      batch_size, seq_len, hidden_dim = real.size()

      alpha = torch.rand(batch_size, 1, 1).to(self.device)
      alpha = alpha.expand(batch_size, seq_len, hidden_dim)

      interpolates = alpha * real + (1 - alpha) * fake
      interpolates.requires_grad_(True)

      if self.conditional:
          C_embed = self.cond_emb(self.C_time_mb, self.C_static_mb)
          d_interpolates = self.netd(interpolates, C_embed)
      else:
          d_interpolates = self.netd(interpolates)


      grad_outputs = torch.ones_like(d_interpolates)

      gradients = torch.autograd.grad(
          outputs=d_interpolates,
          inputs=interpolates,
          grad_outputs=grad_outputs,
          create_graph=True,
          retain_graph=True,
          only_inputs=True
      )[0]

      gradients = gradients.reshape(batch_size, -1)
      gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

      return gp





    def backward_d(self):
      #here we reeplace entirly the funciton
      """
    Hinge loss para el discriminador:

      L_D = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))] + w_gamma * E[max(0, 1 + D(fake_e))]

    Donde:
      - Y_real = D(H)               (features reales)
      - Y_fake = D(H_hat)           (features generadas por G->S)
      - Y_fake_e = D(E_hat)         (features generadas solo por G)
    """
   
      # WGAN critic loss
      loss_real = -self.Y_real.mean()
      loss_fake = self.Y_fake.mean()
      loss_fake_e = self.Y_fake_e.mean()

      # Gradient Penalty in latent space H_hat
      gp = self.gradient_penalty(self.H.detach(), self.H_hat.detach())

      self.err_d = loss_real + loss_fake + self.opt.w_gamma * loss_fake_e + self.opt.gp_lambda * gp

      # Backprop
      self.err_d.backward(retain_graph=True)

      # 🔥 Strong gradient clipping to prevent critic explosion
      torch.nn.utils.clip_grad_norm_(self.netd.parameters(), max_norm=1.0)



      '''
      self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
      self.err_d_fake = self.l_bce(self.Y_fake, torch.zeros_like(self.Y_fake))
      self.err_d_fake_e = self.l_bce(self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
      self.err_d = self.err_d_real + 
                   self.err_d_fake + 
                   self.err_d_fake_e * self.opt.w_gamma
      if self.err_d > 0.15:
        self.err_d.backward(retain_graph=True)'''


    # -----------------------
    # Optimize wrappers (same as original)
    # -----------------------
    def optimize_params_er(self):

      self.forward_er()
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()

      #this line was added to debugging
      self.loss_er = self.l_mse(self.X_tilde, self.X)

      self.loss_er.backward(retain_graph=True)
      self.optimizer_e.step()
      self.optimizer_r.step()

      # NEW: LR schedulers
      self.scheduler_e.step()
      self.scheduler_r.step()

      #print(f"[DEBUG] Loss this iteration: {self.loss_er.item():.6f}")
      return float(self.loss_er.item())

    def optimize_params_er_(self):
      self.forward_er()
      self.forward_s()
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er_()
      self.optimizer_e.step()
      self.optimizer_r.step()
       # NEW
      self.scheduler_e.step()
      self.scheduler_r.step()

    def optimize_params_s(self):
      self.forward_e()
      self.forward_s()
      self.optimizer_s.zero_grad()
      self.backward_s()
      self.optimizer_s.step()
      # NEW
      self.scheduler_s.step()
      return float(self.err_s.item())   # devolvemos el valor numérico

    def optimize_params_g(self):
      self.forward_e()
      self.forward_s()
      self.forward_g()
      self.forward_sg()
      self.forward_rg()
      self.forward_dg()
      self.optimizer_g.zero_grad()
      self.optimizer_s.zero_grad()
      self.backward_g()
      self.optimizer_g.step()
      self.optimizer_s.step()
      self.scheduler_g.step()
      self.scheduler_s.step()

      return float(self.err_g.item())    #  devolvemos G-loss

    def optimize_params_d(self):
      self.forward_e()
      self.forward_g()
      self.forward_sg()
      self.forward_d()
      self.forward_dg()
      self.optimizer_d.zero_grad()
      self.backward_d()
      self.optimizer_d.step()
      # NEW
      self.scheduler_d.step()
      return float(self.err_d.item())
