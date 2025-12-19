"""
TimeGAN model adapted for multivariate vibration signals (accelerometers, RPM, temperature, etc.) - JORFAN BALDOCEDA

Reference:
Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," NeurIPS, 2019.

Adapted version by [Tu Nombre]
Date: [coloca la fecha actual]

This module defines:
(1) Encoder
(2) Recovery
(3) Generator
(4) Supervisor
(5) Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.init as init
#We add this line
from torch.nn.utils import spectral_norm


# -------------------------------
# Weights initialization
# -------------------------------
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


# ============================================
# 🔥 CONDITION EMBEDDING NETWORK
# ============================================
class CondEmbedding(nn.Module):
    def __init__(self, c_time_dim, c_static_dim, hidden_dim):
        super().__init__()
        
        input_dim = c_time_dim + c_static_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, C_time, C_static):
        """
        C_time:   [B, T, c_time_dim]
        C_static: [B, 1, c_static_dim] or [B, T, c_static_dim]
        Returns:
            C_embed: [B, T, hidden_dim]
        """
        B, T, _ = C_time.size()

        # Expand static conditions along time
        if C_static.size(1) == 1:
            C_static = C_static.repeat(1, T, 1)

        cond = torch.cat([C_time, C_static], dim=2)  # [B, T, c_time_dim + c_static_dim]

        return self.net(cond)  # [B, T, hidden_dim]


# -------------------------------
# 1. Encoder
# -------------------------------
class Encoder(nn.Module):
    """
    Embedding network: from original space → latent space
    Input:  X (seq_len, batch, z_dim)
    Output: H (seq_len, batch, hidden_dim)
    """
    def __init__(self, opt):
        super(Encoder, self).__init__()
        
        self.rnn1 = nn.GRU(
            input_size=opt.input_dim,     # ← AHORA USAMOS TODA LA ENTRADA CONCATENADA
            hidden_size=opt.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        
        self.rnn2 = nn.GRU(input_size=opt.hidden_dim,
                   hidden_size=opt.hidden_dim,
                   num_layers=1,
                   batch_first=True)


        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)

        self.ln = nn.LayerNorm(opt.hidden_dim)   # NEW
        
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, x, sigmoid=True):
        # First GRU
        #with torch.backends.cudnn.flags(enabled=False):
        h1, _ = self.rnn1(x)

        # Dropout
        h1 = self.dropout(h1)

        # Second GRU
        #with torch.backends.cudnn.flags(enabled=False):
        h2, _ = self.rnn2(h1)

        # Dense
        H = self.fc(h2)

        # LayerNorm
        H = self.ln(H)

        return H


# -------------------------------
# 2. Recovery
# -------------------------------
class Recovery(nn.Module):
    """
    Recovery network: latent → original feature space
    Input:  H (seq_len, batch, hidden_dim)
    Output: X_tilde (seq_len, batch, z_dim)
    """
    def __init__(self, opt):
        super(Recovery, self).__init__()

        # Las entradas aquí son H, que tiene dim = hidden_dim
        self.rnn1 = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.rnn2 = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc_mid = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.ln = nn.LayerNorm(opt.hidden_dim)

        # 🔥 SALIDA EN ESPACIO DE LOS ACELERÓMETROS (x_dim = 4)
        self.fc = nn.Linear(opt.hidden_dim, opt.x_dim)

        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, h, sigmoid=True):
        #with torch.backends.cudnn.flags(enabled=False):
        h1, _ = self.rnn1(h)
        h1 = self.dropout(h1)

        #with torch.backends.cudnn.flags(enabled=False):
        h2, _ = self.rnn2(h1)

        h2 = self.fc_mid(h2)
        h2 = self.ln(h2)

        X_tilde = self.fc(h2)

        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)

        return X_tilde



# -------------------------------
# 3. Generator
# -------------------------------
class Generator(nn.Module):
    """
    Generator: noise → latent space
    Input:  Z (seq_len, batch, z_dim)
    Output: E (seq_len, batch, hidden_dim)
    """
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        self.rnn = nn.GRU(
            input_size = opt.z_dim + opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=1,
            batch_first=True
        )


        self.dropout = nn.Dropout(0.2)
      
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, Z, C_embed=None):
        if C_embed is None:
            raise ValueError("Generator: C_embed es obligatorio en TimeGAN condicional.")

        """
        Z:       [B, T, z_dim]
        C_embed: [B, T, hidden_dim]
        """
        z_full = torch.cat([Z, C_embed], dim=2)

        #with torch.backends.cudnn.flags(enabled=False):
        g_outputs, _ = self.rnn(z_full)

        g_outputs = self.dropout(g_outputs)
        E = self.fc(g_outputs)
        return E





# -------------------------------
# 4. Supervisor
# -------------------------------
class Supervisor(nn.Module):
    """
    Supervisor: predict next latent sequence from current
    Input:  H (seq_len, batch, hidden_dim)
    Output: S (seq_len, batch, hidden_dim)
    """
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, h, sigmoid=True):
        #with torch.backends.cudnn.flags(enabled=False):
        s_outputs, _ = self.rnn(h)
        s_outputs = self.dropout(s_outputs)
        S = self.fc(s_outputs)
        return S



# -------------------------------
# ⚖️ 5. Discriminator (CONDITIONAL)
# -------------------------------
class Discriminator(nn.Module):
    """
    Conditional WGAN-GP Critic
    Input:  H, C_time, C_static
    Output: critic score (no sigmoid)
    """
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.hidden_dim = opt.hidden_dim
        self.c_time_dim = opt.c_time_dim
        self.c_static_dim = opt.c_static_dim
        self.conditional = opt.conditional

        if self.conditional:
            d_input_dim = self.hidden_dim + self.hidden_dim

        else:
            d_input_dim = self.hidden_dim

        # GRU recibe ahora la concatenación completa
        self.rnn = nn.GRU(
            input_size=d_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.fc_mid  = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln      = nn.LayerNorm(self.hidden_dim)
        self.fc      = nn.Linear(self.hidden_dim, 1)

        self.apply(_weights_init)

    def forward(self, H, C_embed=None):

        """
        H:        [B, T, hidden_dim]
        C_time:   [B, T, c_time_dim]
        C_static: [B, 1, c_static_dim] or [B, T, c_static_dim]
        """

        # 🔥 OPTIONAL: Instance noise (anti-overfitting)
        if getattr(self, "instance_noise", False):
            noise_H = torch.randn_like(H) * 0.01
            H = H + noise_H

            noise_C = torch.randn_like(C_embed) * 0.01 if C_embed is not None else None
            if noise_C is not None:
                C_embed = C_embed + noise_C


        if self.conditional:
            # C_embed ya viene con shape [B, T, hidden_dim]
            D_in = torch.cat([H, C_embed], dim=2)
        else:
            D_in = H


        # 🔥 FIX: disable CuDNN to allow double backward for WGAN-GP
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.rnn(D_in)

        
        out = self.dropout(out)
        out = self.fc_mid(out)
        out = self.ln(out)
        logits = self.fc(out)

        return logits

        
    ''' self.fc = nn.Linear(opt.hidden_dim, 1)  # <- corregido para probabilidad
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, h, sigmoid=True):
        d_outputs, _ = self.rnn(h)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat'''