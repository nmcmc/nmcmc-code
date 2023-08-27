#!/usr/bin/env python3
"""Memory usage analysis for Schwinger model

This script checks amount  memory allocated while evaluating a loss function for Schwinger model.
"""

import argparse
import sys

import torch

import utils.scripts as scripts
import utils.profile
import training.loss
import phys_models.schwinger as sch
from normalizing_flow.gauge_equivariant import make_schwinger_model

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--device', type=str, default='cuda', help='Device to run the training on.')
parser.add_argument('--list-cuda-devices', action='store_true', help="Lists cuda devices and exists")
parser.add_argument('-L', '--lattice-size', type=int, action='store', default=8,
                    help='L - lattice size will be LxL')
parser.add_argument('--batch-size', type=int, action='store', default='256', help='Size of the batch')
parser.add_argument('--loss', default='REINFORCE', choices=['REINFORCE', 'rt'], help='Loss function')
parser.add_argument('--float-dtype', default='float32', choices=['float32', 'float64'],
                    help='Float precision used for training')
parser.add_argument('-b', '--beta', type=float, default='2.0', help='beta')
parser.add_argument('-k', '--kappa', type=float, default='0.276', help='kappa')
parser.add_argument('-v', '--verbose', type=int, default='1',
                    help='Verbosity level if zero does not print anything.')

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

model = make_schwinger_model(**model_cfg, device=torch_device, verbose=args.verbose)
layers = model['layers']
prior = model['prior']

z = prior.sample_n(batch_size=batch_size)
log_prob_z = prior.log_prob(z)

torch.cuda.reset_peak_memory_stats()

with torch.autograd.graph.saved_tensors_hooks(pack_hook := utils.profile.MemoryPackHook(), utils.profile.unpack_hook):
    l, logq, logp = loss_function(z, log_prob_z, model=model, action=qed_action, use_amp=False)

print(f"{pack_hook.mem_u(b=30):.2f}GB memory used for storing {len(set(pack_hook.ptrs))} tensores in the DAG")
len(set(pack_hook.ptrs))

print(f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB allocated by CUDA")

l.backward()

print(f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB allocated by CUDA after running backward()")
