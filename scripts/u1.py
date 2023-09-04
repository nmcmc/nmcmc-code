#!/usr/bin/env python3
"""Training pure U1 gauge model.

This script trains pure gauge U1 model. It can use  different combinations of the gauge equivariant layers and plaquette
coupling layers as well as loss functions.

There are two possible type of equivariant layers:
- plaq - a simple layer using only plaquettes described in hhttps://arxiv.org/abs/2003.06413
- 2x1  - a layer with different masking patterns and using additionally 2x1 Wilson loops
         as used for Schwinger mode and described in https://arxiv.org/abs/2202.11712
and two possible types of plaquette coupling layers both described in https://arxiv.org/abs/2002.02428
- cs  - circular splines
- ncp - non-compact projection

Together with the loss functions, there are 6 possible combinations
(non-compact projection is not compatible with REINFORCE loss)

After training, it can generate samples and compute the free energy that can be compared to the exact value.
The errors are calculated using bootstrap resampling that can be controlled by the parameters
'--n-boot-samples' and '--bin-size'.
"""

import argparse
import time
import sys

import numpy as np
import torch

import training.loss
from training.train import train_step

import utils
import utils.metrics as metrics
import utils.checkpoint as chck
import utils.scripts as scripts

import phys_models.U1 as u1
import normalizing_flow.flow as nf
from normalizing_flow.gauge_equivariant import make_schwinger_model, make_u1_rs_model, make_u1_nc_model, \
    make_u1_nc_model_2x1
from utils.stats_utils import torch_bootstrapf

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=scripts.RawTextArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=str, default='cuda',
                    help='''Device to run the training on.''')
parser.add_argument('--list-cuda-devices', action='store_true',
                    help="Lists cuda devices and exists")
parser.add_argument('-L', '--lattice-size', type=int, action='store', default=8,
                    help='L - lattice size will be LxL')
parser.add_argument('--batch-size', type=int, action='store', default='768',
                    help='Size of the batch')
parser.add_argument('--n-batches', '-n', type=int, default='1',
                    help='Number of batches used for one gradient update')

parser.add_argument('-v', '--verbose', type=int, default='1', help='Verbosity level if zero does not print anything.')
parser.add_argument('--loss', default='REINFORCE', choices=['REINFORCE', 'rt'],
                    help='Loss function')
parser.add_argument('--float-dtype', default='float32', choices=['float32', 'float64'],
                    help='Float precision used for training')
parser.add_argument('--coupling', default='cs', choices=['cs', 'ncp'],
                    help='''Type of plaquettes coupling layer used for the model''')
parser.add_argument('--equiv', default='plaq', choices=['plaq', '2x1', 'sch'],
                    help='''Type of equivariant layers used for the model''')
parser.add_argument('-b', '--beta', type=float, default='1.0', help='beta')
parser.add_argument('-lr', '--learning-rate', type=float, default='0.00025',
                    help='Learning rate for the Adam optimizer')
parser.add_argument('--n-eras', type=int, default='5', help='Number of eras')
parser.add_argument('--n-epochs-per-era', type=int, default=100, help='Numbers of gradient updates per era')
parser.add_argument('--n-samples', type=int, default=2 ** 16, help='Number of samples used for evaluation')
parser.add_argument('--n-boot-samples', type=int, default=100, help='Number of bootstrap samples')
parser.add_argument('--bin-size', type=int, default=16, help='Bin size for bootstrap')

args = parser.parse_args()

if args.list_cuda_devices:
    scripts.list_cuda_devices()
    sys.exit(0)

if args.coupling == 'ncp' and args.loss == 'REINFORCE':
    print("NCP coupling is not compatible with REINFORCE loss")
    sys.exit(0)

if args.equiv == 'sch' and args.coupling == 'cs':
    print('The `sch` equivariant layer is not yet implemented with circular splines coupling layer')
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
link_shape = (2, L, L)
beta = args.beta
u1_action = u1.U1GaugeAction(beta=beta)

loss_function = getattr(training.loss, f"{args.loss}_loss")
model = None
if args.equiv == '2x1':
    if args.coupling == 'cs':
        model_cfg = {'n_layers': 48,
                     'hidden_sizes': [64, 64],
                     'kernel_size': 3,
                     'n_knots': 9,
                     'dilation': [1, 2, 3],
                     'float_dtype': 'float32',
                     'lattice_shape': lattice_shape
                     }
        model = make_schwinger_model(**model_cfg, device=torch_device, verbose=args.verbose)
    elif args.coupling == 'ncp':
        model_cfg = {
            'n_mixture_comps': 6,
            'n_layers': 16,
            'hidden_sizes': [8, 8],
            'kernel_size': 3,
            'lattice_shape': lattice_shape,
            'dilation': 1,
            'float_dtype': 'float32'
        }
        model = make_u1_nc_model_2x1(type='sch_2x1', **model_cfg, device=torch_device, verbose=args.verbose)

elif args.equiv == 'plaq':
    if args.coupling == 'cs':
        model_cfg = {'n_layers': 16,
                     'hidden_sizes': [8, 8],
                     'kernel_size': 3,
                     'n_knots': 9,
                     'dilation': 1,
                     'float_dtype': 'float32',
                     'lattice_shape': lattice_shape
                     }
        model = make_u1_rs_model(**model_cfg, device=torch_device, verbose=args.verbose)
    elif args.coupling == 'ncp':
        model_cfg = {
            'n_mixture_comps': 6,
            'n_layers': 16,
            'hidden_sizes': [8, 8],
            'kernel_size': 3,
            'lattice_shape': lattice_shape,
            'dilation': 1,
            'float_dtype': 'float32'
        }
        model = make_u1_nc_model(type='plaq', **model_cfg, device=torch_device, verbose=args.verbose)
elif args.equiv == 'sch':
    if args.coupling == 'cs':
        model_cfg = {'n_layers': 16,
                     'hidden_sizes': [8, 8],
                     'kernel_size': 3,
                     'n_knots': 9,
                     'dilation': 1,
                     'float_dtype': 'float32',
                     'lattice_shape': lattice_shape
                     }
        model = make_u1_rs_model(type='sch', **model_cfg, device=torch_device, verbose=args.verbose)
    elif args.coupling == 'ncp':
        model_cfg = {
            'n_mixture_comps': 6,
            'n_layers': 16,
            'hidden_sizes': [8, 8],
            'kernel_size': 3,
            'lattice_shape': lattice_shape,
            'dilation': 1,
            'float_dtype': 'float32'
        }
        model = make_u1_nc_model(type='sch', **model_cfg, device=torch_device, verbose=args.verbose)
if not model:
    print(f"Model not defined")
    sys.exit(1)

layers = model['layers']
prior = model['prior']

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
        m = train_step(use_amp=False, model=model, action=u1_action, loss_fn=loss_function, batch_size=args.batch_size,
                       n_batches=args.n_batches, optimizer=optimizer)
        metrics.add_metrics(history, m)
        epochs_done += 1
        if (epoch + 1) % print_freq == 0:
            chck.safe_save_checkpoint(model=layers, optimizer=optimizer, scheduler=None, era=era, model_cfg=model_cfg,
                                      beta=beta, path=f"U1_{args.loss}_{L:02d}x{L:02d}.zip")
            elapsed_time = time.time() - start_time
            avg = metrics.average_metrics(history, args.n_epochs_per_era, history.keys())
            if args.verbose > 1:
                print(f"Finished era {era + 1:d} epoch {epoch + 1:d} elapsed time {elapsed_time:.1f}", end="")
            if epochs_done > 0:
                time_per_epoch = elapsed_time / epochs_done
                time_remaining = (total_epochs - epochs_done) * time_per_epoch
                if args.verbose > 1:
                    print(f"  {time_per_epoch:.2f}s/epoch  remaining time {utils.format_time(time_remaining):s}")
                    metrics.print_dict(avg)

if args.verbose > 0:
    print(f"Elapsed {utils.format_time(elapsed_time)} {elapsed_time / args.n_eras:.2f}s/era")

if args.n_samples > 0:
    if args.verbose > 0:
        print(f"Sampling {args.n_samples} configurations")
    F = -u1.logZ(L, beta)
    u, lq = nf.sample(batch_size=batch_size, n_samples=args.n_samples, prior=prior, layers=layers)
    lp = -u1_action(u)
    lw = lp - lq
    F_q, F_q_std = torch_bootstrapf(lambda x: -torch.mean(x), lw, n_samples=args.n_boot_samples, binsize=args.bin_size)
    if args.verbose > 0:
        print(f"Free energy true={F:.4f} variational={F_q:.4f}+/-{F_q_std:.4f} diff={F_q - F:.4f}")

    F_nis, F_nis_std = torch_bootstrapf(lambda x: -(torch.special.logsumexp(x, 0) - np.log(len(x))), lw,
                                        n_samples=args.n_boot_samples,
                                        binsize=args.bin_size)
    if args.verbose > 0:
        print(f"Free energy true={F:.4f} NIS={F_nis:.4f}+/-{F_nis_std:.4f} diff={F_nis - F:.4f}")
