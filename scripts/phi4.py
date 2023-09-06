#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training phi^4 model

This script trains phi^4 model

After training, it can generate samples and compute the free energy that in case of free field (lambda==0)
can be compared to the exact value. The errors are calculated using bootstrap resampling that can be controlled
by the parameters '--n-boot-samples' and '--bin-size'.
"""

import argparse
import time
import sys
import json

from numpy import log
import torch

import training.loss
from training.train import train_step

import utils
import utils.metrics as metrics
import utils.checkpoint as chck

import phys_models.phi4 as phi4
from normalizing_flow.affine_couplings import make_phi4_affine_layers
import normalizing_flow.flow as nf
import utils.scripts as scripts
from utils.stats_utils import torch_bootstrapf, torch_bootstrapo

parser = argparse.ArgumentParser(description=__doc__, formatter_class=scripts.RawTextArgumentDefaultsHelpFormatter)

parser.add_argument('--device', type=str, default='cuda', help='Device to run the training on.')
parser.add_argument('--list-cuda-devices', action='store_true', help="Lists cuda devices and exists")
parser.add_argument('-L', '--lattice-size', type=int, action='store', default=8,
                    help='L - lattice size will be LxL')
parser.add_argument('--batch-size', type=int, action='store', default='1024', help='Size of the batch')
parser.add_argument('--n-batches', '-n', type=int, default='1',
                    help='Number of batches used for one gradient update')
parser.add_argument('--configuration', action='store', help='Configuration file')
parser.add_argument('-v', '--verbose', type=int, default='1',
                    help='Verbosity level if zero does not print anything.')
parser.add_argument('--loss', default='rt', choices=['REINFORCE', 'rt'], help='Loss function')
parser.add_argument('--float-dtype', default='float32', choices=['float32', 'float64'],
                    help='Float precision used for training')
parser.add_argument('-m2', '--mass2', type=float, default='-4.0',
                    help='m^2 mass squared parameter of the action')
parser.add_argument('-l', '--lambda', type=float, dest='lamda', default='8.0',
                    help='lambda parameter of the action')
parser.add_argument('-lr', '--learning-rate', type=float, default='0.001',
                    help='Learning rate for the Adam optimizer')
parser.add_argument('--n-eras', type=int, default=10, help='Number of eras')
parser.add_argument('--n-epochs-per-era', type=int, default=100,
                    help='Numbers of gradient updates per era')

parser.add_argument('--n-samples', type=int, default=2 ** 16,
                    help='Number of samples used for evaluation')
parser.add_argument('--n-boot-samples', type=int, default=100,
                    help='Number of bootstrap samples')
parser.add_argument('--bin-size', type=int, default=16,
                    help='Bin size for bootstrap')

args = parser.parse_args()

if args.list_cuda_devices:
    scripts.list_cuda_devices()
    sys.exit(0)

if args.verbose > 0:
    print(f"Running on PyTorch {torch.__version__}")

torch_device = args.device
scripts.check_cuda(torch_device)

if args.verbose:
    scripts.describe_device(torch_device)

batch_size = args.batch_size
float_dtype = args.float_dtype

L = args.lattice_size
lattice_shape = (L, L)

action = phi4.ScalarPhi4Action(args.mass2, args.lamda)
loss_function = getattr(training.loss, f"{args.loss}_loss")

model_cfg = {'n_layers': 16,
             'hidden_channels': [16, 16, 16],
             'kernel_size': 3,
             'lattice_shape': lattice_shape
             }

if args.configuration:
    with open(args.configuration) as f:
        model_cfg = json.load(f)

layers = make_phi4_affine_layers(**model_cfg, device=torch_device)
prior = nf.SimpleNormal(torch.zeros(lattice_shape).to(device=torch_device),
                        torch.ones(lattice_shape).to(device=torch_device))

model = {'layers': layers, 'prior': prior}

print_freq = 25  # epochs

history = {
    'dkl': [],
    'std_dkl': [],
    'loss': [],
    'ess': []
}

optimizer = torch.optim.Adam(model['layers'].parameters(), lr=args.learning_rate)

elapsed_time = 0
start_time = time.time()

total_epochs = args.n_eras * args.n_epochs_per_era
epochs_done = 0
if args.verbose > 0:
    print(f"Starting training: {args.n_eras} x {args.n_epochs_per_era} epochs")
for era in range(args.n_eras):
    for epoch in range(args.n_epochs_per_era):
        m = train_step(use_amp=False, model=model, action=action, loss_fn=loss_function, batch_size=args.batch_size,
                       n_batches=args.n_batches, optimizer=optimizer)
        metrics.add_metrics(history, m)
        epochs_done += 1
        if (epoch + 1) % print_freq == 0:
            chck.safe_save_checkpoint(model=layers, optimizer=optimizer, scheduler=None, era=era, model_cfg=model_cfg,
                                      **{'mass2': args.mass2, 'lambda': args.lamda},
                                      path=f"phi4_{args.loss}_{L:02d}x{L:02d}.zip")
            elapsed_time = time.time() - start_time
            avg = metrics.average_metrics(history, args.n_epochs_per_era, history.keys())

            print(f"Finished era {era + 1:d} epoch {epoch + 1:d} elapsed time {elapsed_time:.1f}", end="")
            if epochs_done > 0:
                time_per_epoch = elapsed_time / epochs_done
                time_remaining = (total_epochs - epochs_done) * time_per_epoch
                if args.verbose > 0:
                    print(f"  {time_per_epoch:.2f}s/epoch  remaining time {utils.format_time(time_remaining):s}")
                    metrics.print_dict(avg)

if args.verbose > 0:
    print(f"{elapsed_time / args.n_eras:.2f}s/era")

if args.n_samples > 0:
    print(f"Sampling {args.n_samples} configurations")
    if args.mass2 > 0.0:
        F_exact = phi4.free_field_free_energy(L, args.mass2)
    u, lq = nf.sample(batch_size=batch_size, n_samples=args.n_samples, prior=prior, layers=layers)
    lp = -action(u)
    lw = lp - lq
    F_q, F_q_std = torch_bootstrapf(lambda x: -torch.mean(x), lw, n_samples=args.n_boot_samples, binsize=args.bin_size)

    lw = lp - lq
    F_nis, F_nis_std = torch_bootstrapf(lambda x: -(torch.special.logsumexp(x, 0) - log(len(x))), lw,
                                        n_samples=args.n_boot_samples,
                                        binsize=args.bin_size)
    if args.lamda == 0.0:
        print(f"Variational free energy = {F_q:.3f}+/-{F_q_std:.4f} diff = {F_q - F_exact:.4f}")
        print(f"NIS free energy = {F_nis:.3f}+/-{F_nis_std:.4f} diff = {F_nis - F_exact:.4f}")
    else:
        print(f"Variational free energy = {F_q:.3f}+/-{F_q_std:.4f}")
        print(f"NIS free energy = {F_nis:.3f}+/-{F_nis_std:.4f}")

    mag2, mag2_std = torch_bootstrapo(lambda x: torch.sum(x, dim=(1, 2)) ** 2 / (L * L), u, n_samples=100, binsize=16,
                                      logweights=lw)
    print(f"Magnetization^2 /(L*L) = {mag2.mean():.3f}+/-{mag2_std.mean():.4f}")

