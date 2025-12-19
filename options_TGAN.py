# options_TGAN.py
"""
Options for your Thesis TimeGAN implementation.
Based on:
  - TimeGAN (Yoon et al., NeurIPS 2019)
  - PyTorch adaptation (Zhiwei Zhang, 2021)
"""

import argparse
import os
import torch


class Options():

    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """
     
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="TimeGAN Configuration Options for Rotor System Data"
        )

        # -----------------------------
        #  Data parameters
        # -----------------------------
        self.parser.add_argument(
            '--data_name',
            choices=['sine', 'my_signals'],
            default='my_signals',
            type=str,
            help='Type of dataset to use: sine (synthetic) or my_signals (real signals)'
        )

                # -----------------------------
        #  Conditional TimeGAN options
        # -----------------------------
        self.parser.add_argument(
            '--conditional',
            action='store_true',
            help='Enable conditional TimeGAN (use C_time & C_static)'
        )

        # Dimensiones condicionales (por ahora las ponemos fijas)
        self.parser.add_argument(
            '--c_time_dim',
            type=int,
            default=3,
            help='Dim of time-varying conditions (Motor_current, Speed, Temperature)'
        )

        self.parser.add_argument(
            '--c_static_dim',
            type=int,
            default=2,
            help='Dim of static conditions (weight, distance)'
        )

        # Dim de las series principales (acelerómetros)
        self.parser.add_argument(
            '--x_dim',
            type=int,
            default=4,
            help='Dim of main signals X (accelerometers)'
        )


        self.parser.add_argument('--gp_lambda', 
            type=float, default=10.0, help='Gradient penalty weight')

        self.parser.add_argument('--n_critic', type=int, default=5,
                         help='Number of critic iterations per generator iteration (WGAN-GP)')


        self.parser.add_argument(
            '--seq_len',
            default=256,
            type=int,
            help='Sequence length for each time series window (same as in preprocessing)'
        )
       
        # -----------------------------
        # Model parameters
        # -----------------------------
        self.parser.add_argument(
            '--z_dim',
            default=32,
            type=int,
            help='Dimension of latent vector z'
        )
        self.parser.add_argument(
            '--hidden_dim',
            default=96,
            type=int,
            help='Hidden state dimension in RNN/GRU cells (should be optimized)'
        )
        self.parser.add_argument(
            '--num_layer',
            default=2,
            type=int,
            help='Number of recurrent layers,number of layers (should be optimized)'
        )
        
        # -----------------------------
        # Training parameters
        # -----------------------------
        self.parser.add_argument(
            '--iteration',
            default=50000,
            type=int,
            help='Number of training iterations (should be optimized)'
        )
        self.parser.add_argument(
            '--batch_size',
            default=128,
            type=int,
            help='Number of sequences per mini-batch (should be optimized)'
        )
        self.parser.add_argument(
            '--metric_iteration',
            help='Number of iterations for metric computation',
            default=10,
            type=int
        )

        # -----------------------------
        # Device configuration (same as original)
        # -----------------------------
        self.parser.add_argument('--workers', type=int, help='Number of data loading workers', default=8)
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids: e.g. 0 or 0,1,2. Use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs to use')
        self.parser.add_argument('--model', type=str, default='TimeGAN', help='Model type: TimeGAN')

      # -----------------------------
        # Output folders and experiment info
        # -----------------------------
        self.parser.add_argument('--outf', default='./output', help='Folder to output model checkpoints')
        self.parser.add_argument('--name', type=str, default='experiment_TGAN', help='Name of the experiment')

        # -----------------------------
        # Visualization (original, not used in thesis)
        # -----------------------------
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='Visdom server')
        self.parser.add_argument('--display_port', type=int, default=8097, help='Visdom port')
        self.parser.add_argument('--display_id', type=int, default=0, help='Window id for display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom visualization')

        # -----------------------------
        # Reproducibility and training setup
        # -----------------------------
        self.parser.add_argument('--manualseed', default=-1, type=int, help='Manual random seed')
        self.parser.add_argument('--print_freq', type=int, default=1000, help='Print frequency for logs')
        self.parser.add_argument('--load_weights', action='store_true', help='Load pretrained weights')
        self.parser.add_argument('--resume', default='', help="Path to checkpoints (for resuming training)")

        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # -----------------------------
        # Loss weights (kept from original)
        # -----------------------------
        self.parser.add_argument('--w_gamma', type=float, default=0.5, help='Gamma weight')
        self.parser.add_argument('--w_es', type=float, default=0.1, help='Encoder loss weight')
        self.parser.add_argument('--w_e0', type=float, default=10, help='Embedding loss weight')
        self.parser.add_argument('--w_g', type=float, default=80, help='Generator loss weight')
        self.parser.add_argument('--w_fm', type=float, default=2.0,help='Feature Matching weight for generator')


        # Mark training mode
        self.isTrain = True
        self.opt = None

    def parse(self):
        """Parse arguments (unchanged from original)"""
        self.opt, _ = self.parser.parse_known_args()

        self.opt.isTrain = self.isTrain

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # Set GPU device if applicable
        if self.opt.device == 'gpu' and len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # Set experiment name and folder
        if self.opt.name == 'experiment_TGAN':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.data_name)
        expr_dir = os.path.join(self.opt.outf, self.opt.name)
        
        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)

        
        self.opt.input_dim = None

        # Save options to disk
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.opt