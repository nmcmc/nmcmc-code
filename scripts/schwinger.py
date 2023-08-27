#!/usr/bin/env python3
"""Training Schwinger model

This script trains Schwinger model with two flavours of Wilson fermions based on the paper
https://arxiv.org/abs/2202.11712 and

If n-samples option is set to a non-zero value the script will generate samples and compute the free energy.

It periodically saves the model and optimizer state to a zip file. The model can then be resumed from the checkpoint
using the --continue option.
"""

import argparse
import time
import json
import sys

import torch

import training.loss
from training.train import train_step
import utils
import utils.metrics as metrics
import utils.checkpoint as chck
import utils.scripts as scripts
import phys_models.schwinger as sch
from normalizing_flow.gauge_equivariant import make_schwinger_model
import normalizing_flow.flow as nf
from utils.stats_utils import torch_bootstrapf

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=scripts.RawDescriptionArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=str, default='cuda', help='Device to run the training on.')
parser.add_argument('--list-cuda-devices', action='store_true', help="Lists cuda devices and exists")
parser.add_argument('-L', '--lattice-size', type=int, action='store', default=8,
                    help='L - lattice size will be LxL')
parser.add_argument('--batch-size', type=int, action='store', default='256', help='Size of the batch')
parser.add_argument('--n-batches', '-n', type=int, default='4',
                    help='Number of batches used for one gradient update')
parser.add_argument('--configuration', action='store', help='Configuration file')
parser.add_argument('-v', '--verbose', type=int, default='1',
                    help='Verbosity level if zero does not print anything.')
parser.add_argument('--loss', default='REINFORCE', choices=['REINFORCE', 'rt'], help='Loss function')
parser.add_argument('--float-dtype', default='float32', choices=['float32', 'float64'],
                    help='Float precision used for training')
parser.add_argument('-b', '--beta', type=float, default='2.0', help='beta')
parser.add_argument('-k', '--kappa', type=float, default='0.276', help='kappa')
parser.add_argument('-lr', '--learning-rate', type=float, default='0.00025',
                    help='Learning rate for the Adam optimizer')
parser.add_argument('--n-eras', type=int, default='4', help='Number of eras')
parser.add_argument('--n-epochs-per-era', type=int, default=50,
                    help='Numbers of gradient updates per era')
parser.add_argument('--n-samples', type=int, default=0,
                    help='Number of samples used for evaluation')
parser.add_argument('--n-boot-samples', type=int, default=100,
                    help='Number of bootstrap samples')
parser.add_argument('--bin-size', type=int, default=16,
                    help='Bin size for bootstrap')
parser.add_argument('--continue', dest='cont',
                    help='Continue training from the checkpoint')

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

loss_function = getattr(training.loss, f"{args.loss}_loss")

if not args.cont:
    L = args.lattice_size
    lattice_shape = (L, L)
    link_shape = (2, L, L)
    beta = args.beta
    kappa = args.kappa
    qed_action = sch.QEDAction(beta, kappa)
    model_cfg = {'n_layers': 48,
                 'hidden_sizes': [64, 64],
                 'kernel_size': 3,
                 'n_knots': 9,
                 'dilation': [1, 2, 3],
                 'float_dtype': 'float32',
                 'lattice_shape': lattice_shape
                 }

    if args.configuration:
        with open(args.configuration) as f:
            model_cfg |= json.load(f)

    model = make_schwinger_model(**model_cfg, device=torch_device, verbose=args.verbose)

    optimizer = torch.optim.Adam(model['layers'].parameters(), lr=args.learning_rate)
else:
    loaded = torch.load(args.cont)
    model_cfg = loaded['model_cfg']
    lattice_shape = model_cfg['lattice_shape']
    L = lattice_shape[0]
    beta = loaded['beta']
    kappa = loaded['kappa']
    model = make_schwinger_model(**model_cfg, device=torch_device, verbose=1)
    model['layers'].load_state_dict(loaded['state_dict'])
    qed_action = sch.QEDAction(beta=beta, kappa=kappa)

    optimizer = getattr(torch.optim, loaded['optim'])(model['layers'].parameters())
    optimizer.load_state_dict(loaded['opt_state_dict'])

prior = model['prior']
layers = model['layers']

print_freq = 25  # epochs

history = {
    'dkl': [],
    'std_dkl': [],
    'loss': [],
    'ess': []
}
elapsed_time = 0
start_time = time.time()

total_epochs = args.n_eras * args.n_epochs_per_era
epochs_done = 0
if args.verbose > 0:
    if args.cont:
        print(f"Continuing training from  {args.cont} : {args.n_eras} x {args.n_epochs_per_era} epochs")
    else:
        print(f"Starting training: {args.n_eras} x {args.n_epochs_per_era} epochs")

for era in range(args.n_eras):
    for epoch in range(args.n_epochs_per_era):
        m = train_step(use_amp=False, model=model, action=qed_action, loss_fn=loss_function, batch_size=args.batch_size,
                       n_batches=args.n_batches, optimizer=optimizer)
        metrics.add_metrics(history, m)
        epochs_done += 1
        if (epoch + 1) % print_freq == 0:
            chck.safe_save_checkpoint(model=layers, optimizer=optimizer, scheduler=None, era=era, model_cfg=model_cfg,
                                      beta=beta, kappa=kappa,
                                      path=f"schwinger_{args.loss}_{L:02d}x{L:02d}.zip")
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

    u, lq = nf.sample(batch_size=batch_size, n_samples=args.n_samples, prior=prior, layers=layers)
    lp = -qed_action(u)
    lw = lp - lq
    F_q, F_q_std = torch_bootstrapf(lambda x: -torch.mean(x), lw, n_samples=args.n_boot_samples, binsize=args.bin_size)
    print(f"Variational free energy = {F_q:.2f}+/-{F_q_std:.3f} ")
    lw = lp - lq
    F_nis, F_nis_std = torch_bootstrapf(lambda x: -(torch.special.logsumexp(x, 0) - torch.log(len(x))), lw,
                                        n_samples=args.n_boot_samples,
                                        binsize=args.bin_size)
    print(f"NIS free energy = {F_nis:.2f}+/-{F_nis_std:.3f}")
