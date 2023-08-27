#!/usr/bin/env python3
""" Analyzing Schwinger model

This script loads a model and generates samples from it. It  uses those samples to compute the free energy.
Then it generates samples using Neural Markov Chain Monte-Carlo and computes the topological susceptibility
and chiral condensate.
Errors are computed using bootstrap resampling. The parameters of the bootstrap can be controlled by the
--n-boot-samples and --bin-size options. Bin size should be chosen bigger than the autocorrelation time.
"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np

import normalizing_flow.flow as nf
import utils
import utils.scripts as scripts
import phys_models.U1 as u1
import phys_models.schwinger as sch
from normalizing_flow.gauge_equivariant import make_schwinger_model
from utils.stats_utils import torch_bootstrapf, torch_bootstrapo, ac_and_tau_int
from monte_carlo.nmcmc import metropolize

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=scripts.RawDescriptionArgumentDefaultsHelpFormatter)
parser.add_argument('input', action='store', help='Input file containing the model')
parser.add_argument('-n', '--n-samples', type=int, default=f'{2 ** 16}',
                    help='Number of samples to generate')
parser.add_argument('--batch-size', type=int, default='256',
                    help='Size of the batch. Amount of configurations generated in a single call to the GPU.')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device on which configurations will be generated.')
parser.add_argument('--list-cuda-devices', action='store_true', help="Lists cuda devices and exists")
parser.add_argument('-v', '--verbose', type=int, default='1',
                    help='Verbosity level if zero does not print anything.')
parser.add_argument('--n-boot-samples', type=int, default=100, help='Number of bootstrap samples')
parser.add_argument('--bin-size', type=int, default=16, help='Bin size for bootstrap')

args = parser.parse_args()

torch_device = args.device
float_dtype = "float32"

scripts.check_cuda(torch_device)

if args.list_cuda_devices:
    scripts.list_cuda_devices()
    sys.exit(0)

if args.verbose:
    scripts.describe_device(torch_device)

input_path = Path(args.input)
loaded = torch.load(input_path)

model = make_schwinger_model(**loaded['model_cfg'], device=torch_device, verbose=1)

prior = model['prior']
layers = model['layers']

layers.load_state_dict(loaded['state_dict'])
qed_action = sch.QEDAction(beta=loaded['beta'], kappa=loaded['kappa'])

# Sampling
print(f"""
Sampling {args.n_samples} configurations
""")

u, lq = nf.sample(n_samples=args.n_samples, batch_size=args.batch_size, prior=prior, layers=layers)
lp = -nf.calc_action(u, batch_size=args.batch_size, action=qed_action, device=torch_device)

torch.save({'u': u, 'lq': lq, 'lp': lp}, input_path.stem + '_samples.pt')

ess = nf.compute_ess(lp, lq)
print(f'ESS = {ess:.4f}')

# Variation free energy
n_boot_samples = args.n_boot_samples
binsize = args.bin_size

lw = lp - lq
F_q, F_q_std = torch_bootstrapf(lambda x: -torch.mean(x), lw, n_samples=n_boot_samples, binsize=binsize)
print(f"F_q = {F_q:.2f}+/{F_q_std:.2f}")

F_nis, F_nis_std = torch_bootstrapf(lambda x: -(torch.special.logsumexp(x, 0) - np.log(len(x))), lw,
                                    n_samples=n_boot_samples,
                                    binsize=binsize)
print(f"F_NIS = {F_nis:.4f}+/-{F_nis_std:.4f}")

# Neural Markov Chain Monte-Carlo

print(f"""
Running Neural Markov Chain Monte-Carlo
""")

u_p, s_p, s_q, accepted = metropolize(u, lq, lp)
torch.save({'u_p': u_p, 's_p': s_p, 's_q': s_q, 'accepted': accepted}, input_path.stem + '_mc_samples.pt')

print("Accept rate:", utils.grab(accepted).mean())

Q = utils.grab(u1.topo_charge(u_p))

from utils.stats_utils import bootstrap

X_mean, X_err = bootstrap(Q ** 2, n_samples=n_boot_samples, binsize=binsize)

print(f'Topological susceptibility = {X_mean:.2f} +/- {X_err:.2f}')

# Chiral condensate

cond = sch.calc_condensate(u_p, kappa=loaded['kappa'], batch_size=args.batch_size, device=torch_device,
                           float_dtype=torch.complex128)

cond_avg, cond_std = torch_bootstrapf(lambda x: torch.mean(x), cond, n_samples=n_boot_samples, binsize=binsize)
#
print(f'cond = {cond_avg:.4f}+/-{cond_std:.4f}')

tau, ac = ac_and_tau_int(utils.grab(cond))
print(f'Integrated autocorrelation time for chiral condensate = {tau:.1f}')
