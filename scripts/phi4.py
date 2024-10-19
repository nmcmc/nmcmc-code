#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training phi^4 model

This script trains phi^4 scalar field lattice field theory.

After training, it generates samples and computes the free energy that in case of free field (lambda==0)
is compared to the exact value. The errors are calculated using bootstrap resampling. The mean magnetization square
and mean magnetization absolute value are also calculated.

For more information, check the phi4 notebook.
"""

import time
from pathlib import Path

from numpy import log, sqrt
import torch

import neumc

print(f"Running on PyTorch {torch.__version__}")

# The model will run on the GPU if it is available. Otherwise, it will run on the CPU.
if torch.cuda.is_available():
    torch_device = "cuda"
    print(f"Running on {torch.cuda.get_device_name()}")
else:
    torch_device = "cpu"
    print(f"Running on {neumc.utils.cpuinfo.get_processor_name()} CPU")

float_dtype = "float32"
torch_float_dtype = getattr(torch, float_dtype)

OUTPUT_DIR = "out_phi4"
output_dir_path = Path(OUTPUT_DIR)
output_dir_path.mkdir(parents=True, exist_ok=True)

L = 8
lattice_shape = (L, L)

# Physics model parameters
mass2 = 1.25
lamda = 0.0
action = neumc.physics.phi4.ScalarPhi4Action(mass2, lamda)

# Training parameters
batch_size = 2 ** 10
n_batches = 1
n_eras = 4
n_epochs_per_era = 100

# Loss function used in the training. Possible values are 'rt' and 'REINFORCE' and 'path_gradient'
loss = "REINFORCE"
loss_function = getattr(neumc.training.loss, f"{loss}_loss")

# Model configuration
model_cfg = {
    "n_layers": 16,
    "lattice_shape": lattice_shape,
    "hidden_channels": [16, 16, 16],
    "kernel_size": 3,
}

prior = neumc.nf.flow.SimpleNormal(
    torch.zeros(lattice_shape).to(device=torch_device, dtype=torch_float_dtype),
    torch.ones(lattice_shape).to(device=torch_device, dtype=torch_float_dtype),
)

layers = neumc.nf.affine_cpl.make_phi4_affine_layers(
    **model_cfg, device=torch_device, float_dtype=torch_float_dtype
)

model = {"layers": layers, "prior": prior}

# metrics collected during training
history = {
    "dkl": [],  # Kullback-Leibler divergence
    "std_dkl": [],  # standard deviation of the Kullback-Leibler divergence
    "loss": [],
    "ess": [],  # effective sample size
}

optimizer = torch.optim.Adam(model["layers"].parameters(), lr=0.001)

elapsed_time = 0
start_time = time.time()
print_freq = 25  # epochs
total_epochs = n_eras * n_epochs_per_era
epochs_done = 0
print(f"Starting training: {n_eras} x {n_epochs_per_era} epochs")
for era in range(n_eras):
    for epoch in range(n_epochs_per_era):
        m = neumc.training.train.train_step(
            use_amp=False,
            model=model,
            action=action,
            loss_fn=loss_function,
            batch_size=batch_size,
            n_batches=n_batches,
            optimizer=optimizer,
        )
        neumc.utils.metrics.add_metrics(history, m)
        epochs_done += 1
        if (epoch + 1) % print_freq == 0:
            neumc.utils.checkpoint.safe_save_checkpoint(
                model=layers,
                optimizer=optimizer,
                scheduler=None,
                era=era,
                model_cfg=model_cfg,
                **{"mass2": mass2, "lambda": lamda},
                path=f"{OUTPUT_DIR}/phi4_{loss}_{L:02d}x{L:02d}.zip",
            )
            elapsed_time = time.time() - start_time
            avg = neumc.utils.metrics.average_metrics(
                history, n_epochs_per_era, history.keys()
            )

            print(
                f"Finished era {era + 1:d} epoch {epoch + 1:d} elapsed time {elapsed_time:.1f}",
                end="",
            )
            if epochs_done > 0:
                time_per_epoch = elapsed_time / epochs_done
                time_remaining = (total_epochs - epochs_done) * time_per_epoch

                print(
                    f" {time_per_epoch:.2f}s/epoch  remaining time {neumc.utils.format_time(time_remaining):s}"
                )
                neumc.utils.metrics.print_dict(avg)

print(f"{elapsed_time / n_eras:.2f}s/era")

# Sampling from the trained model

n_samples = 2 ** 17
n_boot_samples = 100  # number of bootstrap samples
bin_size = 1  # Bin size for bootstrap. It is one because the samples are uncorrelated.

if n_samples > 0:
    print(f"Sampling {n_samples} configurations")

    F_exact = 0.0
    # In case of free field we can compare the free energy to the exact value
    if mass2 > 0.0 and lamda == 0.0:
        F_exact = neumc.physics.phi4.free_field_free_energy(L, mass2)

    # This function samples from the trained model. It produces `n_samples` configurations but sampling is done in
    # batches of size `batch_size`. This is because `n_samples` configurations can be too large to fit in the GPU
    # memory.
    u, log_q = neumc.nf.flow.sample(
        batch_size=batch_size, n_samples=n_samples, prior=prior, layers=layers
    )
    log_p = -action(u)
    log_w = log_p - log_q

    # Computes the variational free energy using. It does not correct for the difference between the target Boltzmann
    # distribution and the distribution produced by the model.
    F_q, F_q_std = neumc.utils.stats_utils.torch_bootstrap(
        -log_w, n_samples=n_boot_samples, binsize=bin_size
    )

    # Computes the free energy using the neural importance sampling (NIS) estimator. It corrects for the difference
    # between the target Boltzmann distribution and the distribution produced by the model.
    F_nis, F_nis_std = neumc.utils.stats_utils.torch_bootstrapf(
        lambda x: -(torch.special.logsumexp(x, 0) - log(len(x))),
        log_w,
        n_samples=n_boot_samples,
        binsize=bin_size,
    )
    if lamda == 0.0:
        print(
            f"Variational free energy = {F_q:.3f}+/-{F_q_std:.4f} F_q-F_exact = {F_q - F_exact:.4f}"
        )
        print(
            f"NIS free energy         = {F_nis:.3f}+/-{F_nis_std:.4f} F_q-F_exact = {F_nis - F_exact:.4f}"
        )
        if torch.abs(F_nis - F_exact) > 4 * F_nis_std:
            print("NIS free energy is not consistent with the exact value.")
    else:
        print(f"Variational free energy = {F_q:.3f}+/-{F_q_std:.4f}")
        print(f"NIS free energy         = {F_nis:.3f}+/-{F_nis_std:.4f}")

    #   <M**2>/(L*L)

    mag2, mag2_std = neumc.utils.stats_utils.torch_bootstrap(
        torch.sum(u, dim=(1, 2)) ** 2 / (L * L),
        n_samples=n_boot_samples,
        binsize=bin_size,
        logweights=log_w,
    )

    if lamda == 0.0:
        mag2_exact = 1 / mass2
        print(f"<M^2> /({L}*{L}) = {mag2:.3f}+/-{mag2_std:.4f}")
        if torch.abs(mag2 - mag2_exact) > 4 * mag2_std:
            print(f"<M^2>(L*L) is not consistent with the exact value {mag2_exact}")
    else:
        print(f"<M^2> /({L}*{L}) = {mag2:.3f}+/-{mag2_std:.4f}")

    #  <|M|>/(L*L)

    mag_abs, mag_abs_std = neumc.utils.stats_utils.torch_bootstrap(
        torch.abs(torch.sum(u, dim=(1, 2))) / (L * L),
        n_samples=n_boot_samples,
        binsize=bin_size,
        logweights=log_w,
    )
    if lamda == 0.0:
        mag_abs_exact = sqrt(2 / torch.pi) / L * 1 / sqrt(mass2)
        print(f"<|M|> /({L}*{L}) = {mag_abs:.3f}+/-{mag_abs_std:.4f}")
        if torch.abs(mag_abs - mag_abs_exact) > 4 * mag_abs_std:
            print(f"<|M|>/(L*L) is not consistent with the exact value {mag_abs_exact}")
    else:
        print(f"<|M|> /({L}*{L}) = {mag_abs:.3f}+/-{mag_abs_std:.4f}")
