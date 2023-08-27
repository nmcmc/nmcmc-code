#!/usr/bin/env python3
"""Detailed timing measurements for loss functions.

This script measures the time of the loss function evaluation and the time of the backward pass.
It first performs  several warmup steps and then measures the time of the loss function evaluation and the backward pass
for a given number of measurements reporting the mean and the standard deviation of the measurements.

The model parameters can be specified in a JSON file. The default parameters are used if no file is specified.
"""

import argparse
import torch
import numpy as np
import sys
import json

import utils.scripts as scripts
import training.loss as loss
from normalizing_flow.gauge_equivariant import make_schwinger_model
import phys_models.schwinger as sch

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=str, default='cuda', help='Device to run the training on.')
parser.add_argument('--list-cuda-devices', action='store_true', help="Lists cuda devices and exists")
parser.add_argument('-L', '--lattice-size', type=int, action='store', default='16', help='L - lattice size will be LxL')
parser.add_argument('--batch-size', type=int, action='store', default='256', help='Size of the batch')
parser.add_argument('-b', '--beta', type=float, default='2.0', help='beta')
parser.add_argument('-k', '--kappa', type=float, default='0.276', help='kappa')
parser.add_argument('--configuration', action='store', help='Model parameters configuration JSON file)')
parser.add_argument('--warming', type=int, default='5', help='Number of warmup steps')
parser.add_argument('--measurements', type=int, default='25', help='Number of time measurements')
parser.add_argument('--include-prior-generation', action='store_true',
                    help='Include prior configurations generation in the timing')
parser.add_argument('-v', '--verbose', type=int, default='1', help='Verbosity level. Ff zero does not print anything.')
parser.add_argument('--loss', default='REINFORCE', choices=['REINFORCE', 'rt'], help='Loss function')
args = parser.parse_args()

if args.list_cuda_devices:
    scripts.list_cuda_devices()
    sys.exit(0)

if args.verbose > 0:
    print(f"Running PyTorch {torch.__version__:s}")

torch_device = args.device
scripts.check_cuda(torch_device)

if args.verbose:
    scripts.describe_device(torch_device)

batch_size = args.batch_size

L = args.lattice_size
lattice_shape = (L, L)
beta = args.beta
kappa = args.kappa

qed_action = sch.QEDAction(beta, kappa)

model_cfg = {'n_layers': 48,
             'hidden_sizes': [64, 64],
             'kernel_size': 3,
             'n_knots': 7,
             'dilation': [1, 2, 3],
             'float_dtype': 'float32'
             }

if args.configuration:
    with open(args.configuration) as f:
        cfg = json.load(f)

model_cfg |= cfg

model = make_schwinger_model(lattice_shape=(L, L), **model_cfg, device=torch_device)

layers = model['layers']

prior = model['prior']

loss_fn = getattr(loss, f"{args.loss}_loss")

z = prior.sample_n(batch_size)
log_prob_z = prior.log_prob(z)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
loss_starter, loss_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
back_starter, back_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings = np.zeros((args.measurements, 3))
if args.verbose > 0:
    print("warming ... ")
for i in range(args.warming):
    if args.verbose > 0:
        if i % 5 == 0:
            print(i + 1)
    if args.include_prior_generation:
        z = prior.sample_n(batch_size)
        log_prob_z = prior.log_prob(z)
    l, logq, logp = loss_fn(z, log_prob_z, model=model, action=qed_action, use_amp=False)
    l.backward()
if args.verbose > 0:
    print("measuring ... ")

# WAIT FOR GPU SYNC
torch.cuda.synchronize()
for r in range(args.measurements):
    if args.verbose > 0:
        if r % 5 == 0:
            print(r + 1)
    starter.record()
    loss_starter.record()
    if args.include_prior_generation:
        z = prior.sample_n(batch_size)
        log_prob_z = prior.log_prob(z)
    l, logq, logp = loss_fn(z, log_prob_z, model=model, action=qed_action, use_amp=False)
    loss_ender.record()
    back_starter.record()
    l.backward()
    back_ender.record()
    ender.record()

    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    timings[r, 0] = starter.elapsed_time(ender)
    timings[r, 1] = loss_starter.elapsed_time(loss_ender)
    timings[r, 2] = back_starter.elapsed_time(back_ender)


def parameters_string(args):
    return f"L {args.lattice_size} batch-size {args.batch_size} loss {args.loss}"


print(parameters_string(args), " tot. {0:.2f}+/-{3:.3f} loss  {1:.2f}+/-{4:.3f} back. {2:.2f}+/-{5:.3f}".format(
    *(timings.mean(0) / 1000),
    *(timings.std(0) / 1000)), flush=True)
